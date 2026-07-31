param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path,

    [string]$ApiUrl = "http://localhost:8000/api/v1",

    [int]$PollSeconds = 2,

    [int]$TimeoutMinutes = 60
)

$ErrorActionPreference = "Stop"
$resolvedPath = (Resolve-Path -LiteralPath $Path).Path
$isArchive = [System.IO.Path]::GetExtension($resolvedPath).ToLowerInvariant() -eq ".zip"
$endpoint = if ($isArchive) { "$ApiUrl/process/archive" } else { "$ApiUrl/process" }

Write-Host "Checking backend health..."
$health = & curl.exe -fsS ($ApiUrl.Replace("/api/v1", "/health/live"))
if ($LASTEXITCODE -ne 0) {
    throw "Backend health check failed"
}
Write-Host $health

Write-Host "Uploading: $resolvedPath"
$rawResponse = & curl.exe -fsS -X POST $endpoint -F "file=@$resolvedPath"
if ($LASTEXITCODE -ne 0) {
    throw "Upload failed"
}
$response = $rawResponse | ConvertFrom-Json

$tasks = if ($isArchive) {
    @($response.tasks)
} else {
    @($response)
}
if ($tasks.Count -eq 0) {
    throw "The API accepted no video tasks"
}

Write-Host "Accepted tasks: $($tasks.Count)"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$pending = @{}
foreach ($task in $tasks) {
    $pending[[string]$task.task_id] = $task
}

$resultDirectory = Join-Path (Get-Location) "smoke-results"
New-Item -ItemType Directory -Force -Path $resultDirectory | Out-Null

while ($pending.Count -gt 0) {
    if ((Get-Date) -ge $deadline) {
        throw "Smoke test timed out after $TimeoutMinutes minutes"
    }

    foreach ($taskId in @($pending.Keys)) {
        $status = (& curl.exe -fsS "$ApiUrl/status/$taskId") | ConvertFrom-Json
        Write-Host (
            "{0}: stage={1}, progress={2:N1}%, chunks={3}/{4}" -f
            $taskId,
            $status.stage,
            [double]$status.progress,
            [int]$status.tracking_completed_chunks,
            [int]$status.total_chunks
        )

        if ($status.stage -eq "failed") {
            throw "Task $taskId failed: $($status.error_code): $($status.error_detail)"
        }
        if ($status.stage -eq "completed") {
            $result = & curl.exe -fsS "$ApiUrl/results/$taskId"
            $resultPath = Join-Path $resultDirectory "$taskId.json"
            Set-Content -LiteralPath $resultPath -Value $result -Encoding UTF8
            Write-Host "Saved result: $resultPath"
            $pending.Remove($taskId)
        }
    }

    if ($pending.Count -gt 0) {
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-Host "Smoke test completed successfully for all tasks."
