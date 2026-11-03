# ADRの一覧

数字と方針の根拠は、却下した案とともにADRに残している。書き方は[0000-template.md](0000-template.md)。
ADRの中の「計画書」は、2026-08〜10に書いた非公開の計画メモを指す（リポジトリには含めていない）。

| ADR | 決めたこと | 状態 |
| --- | --- | --- |
| [0001](0001-ci-regression-threshold.md) | CIの回帰ゲートのしきい値 | Superseded by 0010（実装しなかった） |
| [0002](0002-equivalence-margin.md) | 許す差を±2.6pt（モデル1段階分の差）に固定する | Accepted |
| [0003](0003-latency-pass-verdict.md) | レイテンシは、MLPerfの8B向け基準を満たすかの3値で判定する | Accepted |
| [0004](0004-expand-case-set.md) | 評価セットを200件に増やす（正解は機械で作り、クエリはLLMが書く） | Accepted |
| [0005](0005-capacity-check.md) | 並行数のセルが成り立つかを、全エンジン共通の容量の判定で確かめる | Accepted |
| [0006](0006-pin-engine-versions.md) | エンジンの版を固定する | Accepted |
| [0007](0007-main-run-layout.md) | 本番の分け方、レイテンシを1台にまとめること、セルの順番 | Accepted |
| [0008](0008-quantization-noninferiority.md) | 量子化の非劣化を確かめ、精度の足切りと台数で構成を選ぶ（第1段階） | Accepted |
| [0009](0009-batching-engines.md) | 連続バッチングの3エンジンを1台で比べ、bf16どうしを互いの基準にした足切りで選ぶ（第2段階） | Accepted |
| [0010](0010-ci-golden-gate.md) | CIの回帰ゲートを、保存済みの8Bの答えに対するゴールデンテストにする（モデルは動かさない） | Accepted |
