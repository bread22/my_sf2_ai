# my_sf2_ai

- Inspired by https://github.com/linyiLYi/street-fighter-ai
- I cannot duplicate his process by following his chat log, my chatGPT gives some other method
- chatGPT even gives different training model every time I start a new chat
- initially it suggest DNQ and the model is not good, I cannot fine tune reward because the information (health/position/etc) are from buffer images
- then I tried again and ask specifically for baseline3 as [linyiLYi](https://github.com/linyiLYi/street-fighter-ai) does
- chatGPT gives a simple yet workable model (for rookies like me)
- I fine tuned the reward a bit and increase the winning rate significantly once it is trained overnight.
