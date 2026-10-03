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
    [switch]$SkipFrontend
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
function Stop-PortListener {
    param([int]$Port)

    # Only ever targets the PID actually bound to this port. Never kills by
    # process name -- an unrelated Streamlit app runs Python on this machine.
    # SilentlyContinue, not try/catch: with $ErrorActionPreference = 'Stop' a
    # caught exception is still written to the error stream, so a transcript
    # or a recording console shows red "TerminatingError" text for what is
    # simply "nothing is listening" -- alarming to read mid-demo.
    $pids = @(
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )

    if (-not $pids) { Write-Ok "port $Port already free"; return }

    foreach ($processId in $pids) {
        try {
            $proc = Get-Process -Id $processId -ErrorAction Stop
            Write-Warn2 "port $Port held by $($proc.ProcessName) (PID $processId) - stopping it"
            Stop-Process -Id $processId -Force -ErrorAction Stop
        } catch {
            Write-Warn2 "could not stop PID ${processId}: $($_.Exception.Message)"
        }
    }

    # Give the socket a moment to actually release before we bind it again.
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 250
        $still = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
        if (-not $still) { Write-Ok "port $Port free"; return }
    }
    Write-Warn2 "port $Port still shows a listener; continuing anyway"
}

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
    Write-Host '       Create it: python -m venv venv ; venv\Scripts\activate ; cd backend ; pip install -e ".[ml,dev,llm]"'
    exit 1
}
Write-Ok "venv python: $Python"

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

# ── free the ports ────────────────────────────────────────────────────
Write-Step 'Freeing ports 8000 and 3000'
Stop-PortListener -Port 8000
if (-not $SkipFrontend) { Stop-PortListener -Port 3000 }

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
    $nextDir = Join-Path $FrontendDir '.next'
    if (Test-Path $nextDir) {
        Remove-Item -Recurse -Force $nextDir -ErrorAction SilentlyContinue
        Write-Ok 'cleared stale .next'
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
