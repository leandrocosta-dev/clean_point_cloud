<#
Corrige drift de tracking (perda de rastreamento durante o scan, que faz a
peca "rotacionar" e duplicar superficies ciclindricas/circulares num angulo
levemente deslocado).

Estrategia (RANSAC fit + filtra duplicata): ajusta cilindros via RANSAC na
nuvem. Quando ha drift, o ajuste fragmenta em varios cilindros pequenos e
inconsistentes em vez de um unico grande -- o maior fragmento (mais pontos
em acordo) e tratado como a "passada boa"; os fragmentos menores (a(s)
passada(s) desalinhada(s)) sao removidos por correspondencia exata de
coordenadas (C2C distance ~0) contra a nuvem original. So pontos realmente
escaneados sao mantidos -- nada e substituido por geometria sintetica.

Funciona bem quando o drift fragmenta o ajuste em um vencedor claro mais
varios fragmentos menores (confirmado visualmente antes de usar -- revise o
log "Cilindros detectados" e, se preciso, ajuste -SupportPoints).
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$InputPly,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$CCExe,

    # Numero minimo de pontos pra RANSAC considerar um cilindro valido.
    # Baixo demais fragmenta tudo; alto demais pode nao achar nada.
    [int]$SupportPoints = 500,

    # Tolerancia de distancia (C2C) pra considerar um ponto da nuvem original
    # como "pertencente" a um cilindro descartado (duplicata exata/quase
    # exata). Pontos sao copias diretas de coordenadas, entao um valor bem
    # pequeno (perto de zero) ja separa bem.
    [double]$DiscardEpsilon = 0.001,

    [switch]$KeepTemp
)

$ErrorActionPreference = "Stop"

function Find-CCExe {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path $Explicit)) { throw "CloudCompare.exe nao encontrado em: $Explicit" }
        return (Resolve-Path $Explicit).Path
    }
    foreach ($c in @("C:\Program Files\CloudCompare\CloudCompare.exe", "C:\Program Files (x86)\CloudCompare\CloudCompare.exe")) {
        if (Test-Path $c) { return $c }
    }
    throw "CloudCompare.exe nao encontrado nos caminhos padrao. Use -CCExe <caminho>."
}

function Get-PlyVertexCount {
    param([string]$Path)
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $buffer = New-Object byte[] 8192
        $read = $stream.Read($buffer, 0, $buffer.Length)
        $text = [System.Text.Encoding]::ASCII.GetString($buffer, 0, $read)
        if ($text -match "element vertex (\d+)") { return [int]$Matches[1] }
        throw "Nao foi possivel ler 'element vertex' do cabecalho PLY: $Path"
    }
    finally { $stream.Close() }
}

function Invoke-CC {
    param($CCExe, [string[]]$CCArgs, [string]$LogDir, [string]$LogPrefix)
    $stdoutLog = Join-Path $LogDir "$LogPrefix.stdout.log"
    $stderrLog = Join-Path $LogDir "$LogPrefix.stderr.log"
    Write-Host "Comando CloudCompare ($LogPrefix):"
    Write-Host (@($CCExe) + $CCArgs -join " ")
    # Start-Process -ArgumentList nao coloca aspas automaticamente em
    # elementos com espaco (diferente de passar args direto pra um exe via &);
    # sem isso, um nome de arquivo com espaco (ex: saidas do RANSAC) quebra
    # no meio.
    $quotedArgs = $CCArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process -FilePath $CCExe -ArgumentList $quotedArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    Get-Content $stdoutLog | Write-Host
    if ((Get-Item $stderrLog).Length -gt 0) { Get-Content $stderrLog | Write-Host }
    if ($p.ExitCode -ne 0) { throw "CloudCompare falhou (exit code $($p.ExitCode)) em '$LogPrefix'. Veja $stdoutLog" }
}

# --- main ---

if (-not (Test-Path $InputPly)) { Write-Error "Arquivo nao encontrado: $InputPly"; exit 1 }

$outputDir = Split-Path $Output -Parent
if ($outputDir -and -not (Test-Path $outputDir)) { New-Item -ItemType Directory -Force -Path $outputDir | Out-Null }

$ccExePath = Find-CCExe -Explicit $CCExe
$inputPath = (Resolve-Path $InputPly).Path
$commonFlags = @("-SILENT", "-NO_TIMESTAMP", "-C_EXPORT_FMT", "PLY")

$stageDir = Join-Path $outputDir "_drift_$([guid]::NewGuid().ToString('N').Substring(0,8))"
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

try {
    $stagedInput = Join-Path $stageDir "input.ply"
    Copy-Item $inputPath $stagedInput

    # 1) RANSAC: ajusta cilindro(s). Com drift, fragmenta em varios; sem
    #    drift, costuma achar 1 cilindro dominante (ou nenhum, se a peca nao
    #    tiver geometria cilindrica -- nesse caso nao ha nada a corrigir aqui).
    $ransacArgs = $commonFlags + @("-AUTO_SAVE", "OFF", "-O", $stagedInput, `
            "-RANSAC", "ENABLE_PRIMITIVE", "CYLINDER", "SUPPORT_POINTS", "$SupportPoints", `
            "OUTPUT_INDIVIDUAL_SUBCLOUDS", "OUT_CLOUD_DIR", $stageDir)
    Invoke-CC -CCExe $ccExePath -CCArgs $ransacArgs -LogDir $stageDir -LogPrefix "1_ransac"

    $shapes = Get-ChildItem $stageDir -Filter "*_CYLINDER_*_cloud.ply"
    if (-not $shapes -or $shapes.Count -eq 0) {
        Write-Host "Nenhum cilindro detectado -- nada para corrigir. Copiando entrada para a saida sem alteracao."
        Copy-Item $inputPath $Output -Force
        exit 0
    }

    $ranked = $shapes | ForEach-Object {
        [PSCustomObject]@{ Path = $_.FullName; Count = (Get-PlyVertexCount $_.FullName) }
    } | Sort-Object -Property Count -Descending

    Write-Host "Cilindros detectados ($($ranked.Count)):"
    for ($i = 0; $i -lt $ranked.Count; $i++) {
        Write-Host ("  [{0}] {1} pontos - {2}" -f $i, $ranked[$i].Count, $ranked[$i].Path)
    }

    if ($ranked.Count -eq 1) {
        Write-Host "So um cilindro consistente encontrado -- sem fragmentacao, nada para descartar. Copiando entrada para a saida sem alteracao."
        Copy-Item $inputPath $Output -Force
        exit 0
    }

    $good = $ranked[0]
    $bad = $ranked | Select-Object -Skip 1
    Write-Host "Referencia (maior/mais consistente): $($good.Path) ($($good.Count) pontos)"
    Write-Host "Descartando $($bad.Count) fragmento(s) menor(es), total $(($bad | Measure-Object Count -Sum).Sum) pontos"

    # 2) Junta os fragmentos "ruins" (a passada desalinhada) num unico arquivo.
    $mergeArgs = $commonFlags + @("-AUTO_SAVE", "OFF")
    foreach ($b in $bad) { $mergeArgs += @("-O", $b.Path) }
    $badMerged = Join-Path $stageDir "bad_fragments.ply"
    $mergeArgs += @("-MERGE_CLOUDS", "-SAVE_CLOUDS", "FILE", $badMerged)
    Invoke-CC -CCExe $ccExePath -CCArgs $mergeArgs -LogDir $stageDir -LogPrefix "2_merge_bad"

    # 3) C2C distance da nuvem original contra os fragmentos ruins + filtro:
    #    descarta pontos que sao copia exata (distancia ~0) de um ponto
    #    presente nos fragmentos ruins. AUTO_SAVE ON porque C2C_DIST /
    #    FILTER_SF exigem 2 nuvens carregadas e -SAVE_CLOUDS FILE com nomes
    #    multiplos tem um bug de parsing no parser de linha de comando do CC.
    $c2cArgs = $commonFlags + @("-AUTO_SAVE", "ON", "-O", $stagedInput, "-O", $badMerged, `
            "-C2C_DIST", "-FILTER_SF", "$DiscardEpsilon", "MAX")
    Invoke-CC -CCExe $ccExePath -CCArgs $c2cArgs -LogDir $stageDir -LogPrefix "3_filter"

    $result = Get-ChildItem $stageDir -Filter "input_C2C_DIST_FILTERED_*.ply" | Select-Object -First 1
    if (-not $result) {
        throw "Nao achei o arquivo filtrado esperado (input_C2C_DIST_FILTERED_*.ply) em $stageDir"
    }

    # -LiteralPath: nomes gerados pelo CC (ex: "..._FILTERED_[0.001_38.9].ply")
    # tem colchetes, que Copy-Item -Path trataria como wildcard (e silenciosamente
    # nao copiaria nada).
    Copy-Item -LiteralPath $result.FullName -Destination $Output -Force
    $finalCount = Get-PlyVertexCount $Output
    Write-Host "OK: drift corrigido, $finalCount pontos salvos em $Output"
}
finally {
    if ($KeepTemp) {
        Write-Host "Staging mantido (-KeepTemp): $stageDir"
    }
    else {
        Remove-Item -Recurse -Force $stageDir -ErrorAction SilentlyContinue
    }
}

exit 0