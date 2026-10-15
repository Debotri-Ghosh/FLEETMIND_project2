<#
.SYNOPSIS
    Bring FleetMind up for a demo with one command.

.DESCRIPTION
    Frees ports 8000 and 3000, starts a clean backend and frontend, waits until
    each actually answers, and prints the demo logins.

    This exists because the manual netstat / taskkill / restart sequence caused
    real problems during development: a stale uvicorn process serving old code
    produced phantom failures more than once, and uvicorn runs without
    --reload so a leftover process is easy to miss. Under defence pressure that
    is not something to remember -- it should be one command.

    Deliberately conservative about what it kills. It only stops processes
    actually listening on 8000 or 3000, never "all python" or "all node" --
    there is an unrelated Streamlit app on this machine that must survive.

.PARAMETER NoReset
    Leave the replay where it is. By default the script rewinds it to zero so
    the live-automation demo starts from a known state.

.PARAMETER SkipFrontend
    Backend only.

.EXAMPLE
    .\scripts\start-demo.ps1

.EXAMPLE
    .\scripts\start-demo.ps1 -NoReset
#>

[CmdletBinding()]
param(
    [switch]$NoReset,
    [switch]$SkipFrontend,
    [switch]$AllowDockerPortConflict
)

$ErrorActionPreference = 'Stop'
# Probes below are allowed to fail; they use -ErrorAction SilentlyContinue so
# that expected failures never reach the console. $Error is cleared at the end
# so a presenter does not see a stale red backlog from the startup probes.

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$BackendDir  = Join-Path $RepoRoot 'backend'
$FrontendDir = Join-Path $RepoRoot 'frontend'
$Python      = Join-Path $RepoRoot 'venv\Scripts\python.exe'
$LogDir      = Join-Path $RepoRoot '.demo-logs'

$ApiUrl = 'http://127.0.0.1:8000'
$UiUrl  = 'http://localhost:3000'

function Write-Step  { param($m) Write-Host "`n>> $m" -ForegroundColor Cyan }
function Write-Ok    { param($m) Write-Host "   OK  $m" -ForegroundColor Green }
function Write-Warn2 { param($m) Write-Host "   !   $m" -ForegroundColor Yellow }
function Write-Fail  { param($m) Write-Host "   X   $m" -ForegroundColor Red }

# ── free a port, carefully ────────────────────────────────────────────
function Get-OurPortListeners {
    <#
        PIDs bound to $Port whose COMMAND LINE identifies them as this
        project's own server. The command line, not the port alone.

        Killing every holder of a port is unsafe on this machine, and that was
        a real defect here: Docker Desktop binds 0.0.0.0:8000 and wslrelay.exe
        binds [::1]:8000, so an earlier version of this function attempted
        `taskkill /T /F` on com.docker.backend.exe every time it ran. Windows
        happened to refuse, but on a machine where it succeeds, starting the
        demo would take Docker Desktop down.

        Those bindings also never needed freeing: the backend binds
        127.0.0.1:8000 specifically and coexists with Docker's 0.0.0.0:8000
        perfectly well -- it did so for an entire working session. So the
        correct behaviour is to leave anything we did not start alone.
    #>
    param([int]$Port, [string]$Match)

    $pids = @(
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    $ours = @()
    $theirs = @()
    foreach ($processId in $pids) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
        if (-not $proc) { continue }
        if ($proc.CommandLine -and $proc.CommandLine -match $Match) {
            $ours += $proc
        } else {
            $theirs += $proc
        }
    }
    return @{ Ours = $ours; Theirs = $theirs }
}

function Stop-PortListener {
    param([int]$Port, [string]$Match)

    # SilentlyContinue, not try/catch: with $ErrorActionPreference = 'Stop' a
    # caught exception is still written to the error stream, so a transcript
    # or a recording console shows red "TerminatingError" text for what is
    # simply "nothing is listening" -- alarming to read mid-demo.
    $found = Get-OurPortListeners -Port $Port -Match $Match

    foreach ($proc in $found.Theirs) {
        Write-Ok "port $Port also held by $($proc.Name) (PID $($proc.ProcessId)) - not ours, left alone"
    }

    if (-not $found.Ours) { Write-Ok "port $Port has no FleetMind listener"; return }

    foreach ($proc in $found.Ours) {
        try {
            Write-Warn2 "port $Port held by our $($proc.Name) (PID $($proc.ProcessId)) - stopping it"
            # taskkill /T, not Stop-Process: `npm run dev` is a process TREE
            # (cmd -> npm -> node "next dev"), and only the outermost one holds
            # the port. Stop-Process kills that one and orphans the `next dev`
            # child, which keeps running and keeps writing to .next. Two
            # orphaned dev servers then fight over the same directory: one
            # writes the chunks, the other serves and 404s every one of them,
            # so the page loads as blank HTML with no JS or CSS.
            & taskkill.exe /PID $proc.ProcessId /T /F 2>&1 | Out-Null
        } catch {
            Write-Warn2 "could not stop PID $($proc.ProcessId): $($_.Exception.Message)"
        }
    }

    # Wait for OUR listener to go, not for the port to be empty: a third-party
    # holder we deliberately left alone would otherwise make this always warn.
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 250
        $still = Get-OurPortListeners -Port $Port -Match $Match
        if (-not $still.Ours) { Write-Ok "port $Port free of FleetMind listeners"; return }
    }
    Write-Warn2 "port $Port still shows a FleetMind listener; continuing anyway"
}

# Command-line patterns identifying this project's own servers. Deliberately
# specific: 'uvicorn' alone would match any uvicorn app on the machine, and
# 'node' alone would match every Node process.
$BackendMatch  = 'uvicorn.*fleetmind\.api\.main'
# Matches every shape of this project's Next process tree, including the
# `start-server.js` child that actually holds port 3000, which is at a
# different path from `next/dist/bin/next`. Keyed on the project directory so
# it survives a Next version bump, and `[\/]+` accepts BACKSLASHES -- the
# previous pattern used `[\/]`, which is forward-slash only and therefore
# could never match a Windows path.
$FrontendMatch = 'fleetmind.frontend.node_modules.*next'   # '.' matches \ and / alike -- no escaping to get wrong

# ── wait for a URL to answer ──────────────────────────────────────────
function Wait-ForUrl {
    param(
        [string]$Url,
        [string]$Name,
        [int]$TimeoutSeconds = 120
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastErr  = $null
    while ((Get-Date) -lt $deadline) {
        # Every failed probe before the server is up is expected. In Windows
        # PowerShell 5.1 Invoke-WebRequest raises a *terminating* error on a
        # refused connection, so -ErrorAction SilentlyContinue does not
        # suppress it and -ErrorActionPreference 'Stop' would abort the whole
        # script -- it has to be try/catch. A caught exception prints nothing
        # on a real console; Start-Transcript does log a "TerminatingError"
        # line for it, which is a transcript artefact, not console output.
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) {
                Write-Ok "$Name responding (HTTP $($r.StatusCode))"
                return $true
            }
        } catch {
            $lastErr = $_.Exception.Message
            # A 404 on a route this project definitely serves is not "not up
            # yet" -- it is a DIFFERENT server answering on this port. Seen for
            # real: Docker Desktop was running a stale, pre-auth FleetMind API
            # container publishing 0.0.0.0:8000, which has no /livez route.
            if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 404) {
                $lastErr = "HTTP 404 -- something other than FleetMind's current $Name is answering on this port (see the Docker warning above)"
            }
        }
        Start-Sleep -Seconds 2
    }
    Write-Fail "$Name did not respond within ${TimeoutSeconds}s. Last error: $lastErr"
    return $false
}

# ── preflight ─────────────────────────────────────────────────────────
Write-Step 'Checking prerequisites'

if (-not (Test-Path $Python)) {
    Write-Fail "Python venv not found at $Python"
    Write-Host '       Create it: python -m venv venv ; venv\Scripts\activate ; cd backend ; pip install -e ".[ml,dev,llm]" -c constraints.txt'
    exit 1
}
Write-Ok "venv python: $Python"

# Docker containers publishing our ports. These are WARNED about, never stopped:
# they belong to whoever started them, and a running compose stack may be
# deliberate. But they must be visible, because a container on 0.0.0.0:8000
# shadows this backend for every `localhost` / [::1] client, and answers on
# 127.0.0.1 until uvicorn has bound -- which looks exactly like a broken app.
$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($docker) {
    $published = @()
    try {
        $published = @(& docker ps --format '{{.Names}}|{{.Image}}|{{.Ports}}' 2>$null |
            Where-Object { $_ -match ':(8000|3000)->' })
    } catch { $published = @() }
    foreach ($row in $published) {
        $parts = $row -split '\|'
        Write-Warn2 "Docker container '$($parts[0])' ($($parts[1])) publishes a FleetMind port: $($parts[2])"
    }
    if ($published) {
        Write-Warn2 'It shadows this demo for every localhost/[::1] client, and on this machine'
        Write-Warn2 "the backend probe below then saw only that container's 404s for 120 s"
        Write-Warn2 'even though uvicorn had bound 127.0.0.1:8000 (cause not yet established).'
        if (-not $AllowDockerPortConflict) {
            Write-Fail 'Refusing to start a demo that cannot be trusted. Containers are never stopped'
            Write-Fail 'by this script -- they belong to whoever started them. Either:'
            Write-Host '         docker compose stop            (reversible: docker compose start)' -ForegroundColor Red
            Write-Host '       or re-run with -AllowDockerPortConflict to proceed anyway.' -ForegroundColor Red
            exit 2
        }
        Write-Warn2 '-AllowDockerPortConflict given: continuing despite the conflict.'
    }
}

if (-not $SkipFrontend) {
    # Node is installed user-scope via winget on this machine, not in Program
    # Files, so report where it actually resolved rather than assuming.
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) {
        Write-Fail 'node not found on PATH. Node 24 is installed user-scope via winget here;'
        Write-Host '       an elevated shell that does not inherit the user PATH will not see it.'
        exit 1
    }
    Write-Ok "node: $($node.Source)"

    if (-not (Test-Path (Join-Path $FrontendDir 'node_modules'))) {
        Write-Warn2 'frontend/node_modules missing - running npm ci (one-off, may take a minute)'
        Push-Location $FrontendDir
        try { & npm ci } finally { Pop-Location }
    }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ── sweep orphaned next dev processes ─────────────────────────────────
# A `next dev` child can survive its parent and keep writing to .next without
# holding port 3000, so freeing the port alone does not find it. Matched on the
# command line so only THIS project's dev servers are touched.
function Assert-PatternCanMatch {
    <#
        A sweep that finds nothing must prove it was capable of finding
        something. The previous version of Stop-OrphanedNextDev matched
        'next[\/]dist[\/]bin[\/]next' -- a character class holding only a
        FORWARD slash -- against Windows command lines full of backslashes. It
        could never match anything, so it printed "no stray next dev processes"
        on every run of its entire existence while sweeping nothing at all.

        This is the fourth check in this project found incapable of firing. The
        standing rule is that a check asserting "nothing is wrong" must also
        assert that it looked at something, so each pattern is verified against
        a known-good sample before its result is trusted.
    #>
    param([string]$Pattern, [string]$Sample, [string]$Name)

    if ($Sample -notmatch $Pattern) {
        Write-Fail "$Name pattern cannot match its own known-good sample -- the sweep is dead"
        Write-Fail "  pattern: $Pattern"
        Write-Fail "  sample : $Sample"
        return $false
    }
    return $true
}

function Stop-StrayFleetMindProcesses {
    <#
        Processes belonging to this project that are NOT holding a port, so
        freeing the port never finds them.

        Two kinds exist in practice. A `next dev` child can outlive its parent
        and keep writing to .next, and two dev servers then fight over that
        directory: one writes the chunks, the other serves and 404s every one,
        so the page renders as blank unstyled HTML. And a `uvicorn
        fleetmind.api.main` process can linger after the port listener is gone,
        ready to grab the port back and serve stale code -- uvicorn runs without
        --reload here, so stale code is invisible until something behaves oddly.

        Matching is on the command line, so the unrelated Streamlit app and the
        other `uvicorn main:app` services on this machine are never touched.
    #>
    $sweeps = @(
        @{ Name = 'next dev';  Pattern = $script:FrontendMatch
           Sample = 'C:\Users\x\Documents\Projects\fleetmind\frontend\node_modules\next\dist\server\lib\start-server.js' }
        @{ Name = 'uvicorn';   Pattern = $script:BackendMatch
           Sample = '"C:\python.exe" -m uvicorn fleetmind.api.main:app --host 127.0.0.1 --port 8000' }
    )

    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                Where-Object { $_.CommandLine })

    foreach ($sweep in $sweeps) {
        if (-not (Assert-PatternCanMatch -Pattern $sweep.Pattern -Sample $sweep.Sample -Name $sweep.Name)) {
            continue
        }
        $strays = @($all | Where-Object { $_.CommandLine -match $sweep.Pattern })
        foreach ($o in $strays) {
            # The process list is a snapshot, and `taskkill /T` kills a whole
            # tree: stopping a `next dev` parent also stops its start-server.js
            # child, which is still in the snapshot. Re-check before each kill,
            # or taskkill fails on the already-dead child -- and in Windows
            # PowerShell 5.1 a native command's stderr under
            # $ErrorActionPreference = 'Stop' is a TERMINATING error, which
            # aborted the whole script with the demo down. Found by running it.
            if (-not (Get-Process -Id $o.ProcessId -ErrorAction SilentlyContinue)) {
                Write-Ok "stray $($sweep.Name) PID $($o.ProcessId) already gone (killed with its parent tree)"
                continue
            }
            Write-Warn2 "stopping stray $($sweep.Name) (PID $($o.ProcessId))"
            try {
                & taskkill.exe /PID $o.ProcessId /T /F 2>&1 | Out-Null
            } catch {
                # Lost a race with its own parent's tree kill -- harmless.
                Write-Ok "stray $($sweep.Name) PID $($o.ProcessId) exited before it could be stopped"
            }
        }
        if (-not $strays) {
            Write-Ok "no stray $($sweep.Name) processes (swept $($all.Count) processes)"
        }
    }
}

# ── free the ports ────────────────────────────────────────────────────
Write-Step 'Freeing ports 8000 and 3000'
Stop-PortListener -Port 8000 -Match $BackendMatch
if (-not $SkipFrontend) {
    Stop-PortListener -Port 3000 -Match $FrontendMatch
}
# After the ports are free, sweep anything of ours that was NOT holding a port
# -- an orphaned `next dev` child or a lingering uvicorn. Freeing a port can
# never find those, and both cause failures that look like something else.
Stop-StrayFleetMindProcesses

# ── backend ───────────────────────────────────────────────────────────
Write-Step 'Starting backend (uvicorn, no --reload)'
$apiLog = Join-Path $LogDir 'backend.log'
Start-Process -FilePath $Python `
    -ArgumentList '-m','uvicorn','fleetmind.api.main:app','--host','127.0.0.1','--port','8000' `
    -WorkingDirectory $BackendDir `
    -RedirectStandardOutput $apiLog `
    -RedirectStandardError (Join-Path $LogDir 'backend.err.log') `
    -WindowStyle Hidden | Out-Null
Write-Ok "launched; log: $apiLog"

# /livez is the unauthenticated probe -- /health needs a token.
if (-not (Wait-ForUrl -Url "$ApiUrl/livez" -Name 'backend' -TimeoutSeconds 120)) {
    Write-Host "`nBackend log tail:" -ForegroundColor Red
    if (Test-Path $apiLog) { Get-Content $apiLog -Tail 20 }
    exit 1
}

# ── frontend ──────────────────────────────────────────────────────────
if (-not $SkipFrontend) {
    Write-Step 'Starting frontend (next dev)'
    $uiLog = Join-Path $LogDir 'frontend.log'

    # A .next directory left by a production build breaks the dev server's
    # asset manifests, and every page 404s its own JS. Cheap to avoid.
    # Must happen AFTER every dev server is dead -- deleting .next underneath a
    # running one leaves a half-written cache that 404s its own chunks.
    $nextDir = Join-Path $FrontendDir '.next'
    if (Test-Path $nextDir) {
        Remove-Item -Recurse -Force $nextDir -ErrorAction SilentlyContinue
        if (Test-Path $nextDir) {
            Write-Warn2 '.next could not be fully removed (a process may still hold it)'
        } else {
            Write-Ok 'cleared stale .next'
        }
    }

    Start-Process -FilePath 'cmd.exe' `
        -ArgumentList '/c','npm run dev' `
        -WorkingDirectory $FrontendDir `
        -RedirectStandardOutput $uiLog `
        -RedirectStandardError (Join-Path $LogDir 'frontend.err.log') `
        -WindowStyle Hidden | Out-Null
    Write-Ok "launched; log: $uiLog"

    if (-not (Wait-ForUrl -Url "$UiUrl/login" -Name 'frontend' -TimeoutSeconds 180)) {
        Write-Host "`nFrontend log tail:" -ForegroundColor Red
        if (Test-Path $uiLog) { Get-Content $uiLog -Tail 20 }
        exit 1
    }
}

# ── reset the replay ──────────────────────────────────────────────────
if (-not $NoReset) {
    Write-Step 'Rewinding the replay to zero'
    try {
        $login = Invoke-RestMethod -Uri "$ApiUrl/auth/login" -Method Post `
            -ContentType 'application/json' -ErrorAction Stop `
            -Body '{"email":"admin@fleetmind.example.com","password":"admin-demo-2026"}'

        $headers = @{ Authorization = "Bearer $($login.access_token)" }
        Invoke-RestMethod -Uri "$ApiUrl/replay/reset" -Method Post -Headers $headers | Out-Null
        $st = Invoke-RestMethod -Uri "$ApiUrl/replay/status" -Headers $headers
        Write-Ok "replay at $($st.released)/$($st.total_windows), ticks $($st.worker.ticks)"
    } catch {
        Write-Warn2 "could not reset the replay: $($_.Exception.Message)"
        Write-Warn2 'Has the database been seeded? python -m fleetmind.db.seed_users'
    }
}

# ── ready ─────────────────────────────────────────────────────────────
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host ' FleetMind is up' -ForegroundColor Green
Write-Host '============================================================' -ForegroundColor Green
Write-Host "  Dashboard : $UiUrl"
Write-Host "  API docs  : $ApiUrl/docs"
Write-Host ''
Write-Host '  Demo logins (passwords are public on purpose)' -ForegroundColor Cyan
Write-Host '  ------------------------------------------------------------'
Write-Host '  admin@fleetmind.example.com               admin-demo-2026     admin'
Write-Host '  operator.a@fleetmind.example.com          operator-demo-2026  operator - Plant A (0 HP)'
Write-Host '  operator.c@fleetmind.example.com          operator-demo-2026  operator - Plant C (2 HP)'
Write-Host '  customer.acme@fleetmind.example.com       customer-demo-2026  customer - Acme Manufacturing'
Write-Host '  customer.northwind@fleetmind.example.com  customer-demo-2026  customer - Northwind Steel'
Write-Host ''
Write-Host '  Start on the Dashboard and press "Start live replay".' -ForegroundColor Cyan
Write-Host "  Logs: $LogDir"
Write-Host ''

# The probe failures above are expected and already handled; clear them so
# the session does not carry a misleading error backlog into the demo.
$Error.Clear()
