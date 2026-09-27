| emotion | top tokens |
|---|---|
| afraid | `IMIT` `Incoming` `incer` `好不容易` (with great difficulty) `ENCHMARK` `istol` `加重` (aggravate) `luxe` |
| angry | `踽` `随时随` `筚` `﹗` `狠抓` (crack down on) `噙` `guit` `蹁` |
| ashamed | `обязатель` `emetery` `唉` (sigh) `reality` `更深` (deeper) `imposed` `羞` (shame) `bersome` |
| calm | `👆` `oplay` `魔龙` (demon dragon) `痄` `蟊` `b'\xb1\x85'` `自然` (natural) `'gc` |
| desperate | `缥` `朋友们对` (friends, towards) `Attempt` `(BitConverter` `holland` `葳` `wors` `Attempts` |
| disgusted | `随时随` `attempted` `不堪` (unbearable) `(withIdentifier` `狠抓` (crack down on) `说出来` (say it out loud) `侮辱` (insult) `/exec` |
| excited | `采矿等` (mining, etc.) `BindingUtil` `discrepan` `激动` (excited) `francais` `AsyncResult` `mù` `兴奋` (excited) |
| joyful | `痄` `harmon` `нарушен` `ör` `Harmon` `halk` `BootApplication` `"','` |
| lonely | `空` (empty) `absence` `vacant` `Empty` `为空` (is empty) `_empty` `.Empty` `isEmpty` |
| proud | `-------------</` `misunder` `@dynamic` `discrepan` `----------</` `瘆` `痄` `nı` |
| sad | `Empty` `空` (empty) `mourn` `isEmpty` `hookers` `lipstick` `深刻的` (profound) `Altern` |
| surprised | `℅` `tín` `ınızı` `无论是其` (whether it) `查看全文` (read the full text) `谆` `慎重` (cautious) `可以更好` (can be better) |

*Logit lens of the 12 emotion vectors of Qwen2.5-Coder-7B-Instruct at layer 19, the steering layer: each vector is passed through the model's final norm and output projection, and the table shows the 8 tokens it most promotes (glosses in brackets; `b'…'` is part of a multi-byte character). At this depth most top tokens are unrelated fragments, many of them Chinese, but some fit their emotion: 激动 and 兴奋 ("excited") for excited, 羞 ("shame") for ashamed, "empty", "absence" and "vacant" for lonely, "mourn" for sad, 侮辱 ("insult") for disgusted. Calm, joyful and proud show no related token in their top 8.*
