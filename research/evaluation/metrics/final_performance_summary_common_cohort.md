# Final common-cohort AUROC results

Generated at: 2026-06-08T12:24:07

Main analysis cohort: target-specific all-model common valid cohort.

## Common cohort size

| Target | Nodes | Patients | Positive | Negative | Excluded from valid union |
|---|---:|---:|---:|---:|---:|
| 1m | 88640 | 1181 | 557 | 88083 | 13 |
| 1y | 76684 | 1069 | 1640 | 75044 | 300 |
| 5y | 35079 | 683 | 1797 | 33282 | 1190 |

## Overall AUROC

### Target 1m

| Model | N | AUROC (95% CI) |
|---|---:|---:|
| XGBoost | 88640 | 0.9252 (0.8991-0.9471) |
| LSTM | 88640 | 0.8853 (0.8202-0.9353) |
| RSF | 88640 | 0.8955 (0.8570-0.9248) |
| Qwen3-4B SFT | 88640 | 0.9121 (0.8832-0.9372) |
| Qwen3-4B baseline | 88640 | 0.7022 (0.6483-0.7465) |
| Llama3.1-8B | 88640 | 0.5350 (0.5133-0.5549) |
| Huatuo-O1-7B | 88640 | 0.5404 (0.5015-0.5754) |

### Target 1y

| Model | N | AUROC (95% CI) |
|---|---:|---:|
| XGBoost | 76684 | 0.8199 (0.7763-0.8653) |
| LSTM | 76684 | 0.7341 (0.6507-0.8087) |
| RSF | 76684 | 0.7731 (0.7178-0.8278) |
| Qwen3-4B SFT | 76684 | 0.7767 (0.7150-0.8356) |
| Qwen3-4B baseline | 76684 | 0.6440 (0.6060-0.6853) |
| Llama3.1-8B | 76684 | 0.5391 (0.5189-0.5574) |
| Huatuo-O1-7B | 76684 | 0.5339 (0.5082-0.5591) |

### Target 5y

| Model | N | AUROC (95% CI) |
|---|---:|---:|
| XGBoost | 35079 | 0.7670 (0.7054-0.8221) |
| LSTM | 35079 | 0.7108 (0.6531-0.7731) |
| RSF | 35079 | 0.7083 (0.6465-0.7728) |
| Qwen3-4B SFT | 35079 | 0.7122 (0.6421-0.7808) |
| Qwen3-4B baseline | 35079 | 0.5707 (0.5334-0.6067) |
| Llama3.1-8B | 35079 | 0.5189 (0.4997-0.5380) |
| Huatuo-O1-7B | 35079 | 0.5088 (0.4938-0.5242) |

## Stage AUROC

### Target 1m

| Stage | Model | N | AUROC (95% CI) |
|---|---|---:|---:|
| Stage1 0d-1m | XGBoost | 19319 | 0.8988 (0.8358-0.9444) |
| Stage1 0d-1m | LSTM | 19319 | 0.8414 (0.7003-0.9322) |
| Stage1 0d-1m | RSF | 19319 | 0.8791 (0.8375-0.9166) |
| Stage1 0d-1m | Qwen3-4B SFT | 19319 | 0.8766 (0.8170-0.9282) |
| Stage1 0d-1m | Qwen3-4B baseline | 19319 | 0.6282 (0.5724-0.6842) |
| Stage1 0d-1m | Llama3.1-8B | 19319 | 0.5009 (0.4636-0.5325) |
| Stage1 0d-1m | Huatuo-O1-7B | 19319 | 0.4967 (0.4435-0.5429) |
| Stage2 2m-3m | XGBoost | 9483 | 0.8768 (0.7509-0.9466) |
| Stage2 2m-3m | LSTM | 9483 | 0.8675 (0.7900-0.9447) |
| Stage2 2m-3m | RSF | 9483 | 0.8655 (0.7203-0.9407) |
| Stage2 2m-3m | Qwen3-4B SFT | 9483 | 0.8942 (0.8081-0.9490) |
| Stage2 2m-3m | Qwen3-4B baseline | 9483 | 0.6048 (0.4261-0.7334) |
| Stage2 2m-3m | Llama3.1-8B | 9483 | 0.5394 (0.4881-0.6085) |
| Stage2 2m-3m | Huatuo-O1-7B | 9483 | 0.5595 (0.4316-0.6709) |
| Stage3 4m-12m | XGBoost | 19167 | 0.8869 (0.8213-0.9380) |
| Stage3 4m-12m | LSTM | 19167 | 0.8125 (0.6561-0.9313) |
| Stage3 4m-12m | RSF | 19167 | 0.7294 (0.6030-0.8459) |
| Stage3 4m-12m | Qwen3-4B SFT | 19167 | 0.8397 (0.7215-0.9381) |
| Stage3 4m-12m | Qwen3-4B baseline | 19167 | 0.4840 (0.3245-0.6460) |
| Stage3 4m-12m | Llama3.1-8B | 19167 | 0.5173 (0.4283-0.6072) |
| Stage3 4m-12m | Huatuo-O1-7B | 19167 | 0.4606 (0.3739-0.5595) |
| Stage4 1y-2y | XGBoost | 12636 | 0.9288 (0.8197-0.9785) |
| Stage4 1y-2y | LSTM | 12636 | 0.9175 (0.7480-0.9864) |
| Stage4 1y-2y | RSF | 12636 | 0.8314 (0.5539-0.9605) |
| Stage4 1y-2y | Qwen3-4B SFT | 12636 | 0.8339 (0.6572-0.9264) |
| Stage4 1y-2y | Qwen3-4B baseline | 12636 | 0.6276 (0.4427-0.7358) |
| Stage4 1y-2y | Llama3.1-8B | 12636 | 0.5786 (0.4629-0.6333) |
| Stage4 1y-2y | Huatuo-O1-7B | 12636 | 0.5814 (0.4625-0.6367) |
| Stage5 >2y | XGBoost | 24329 | 0.9721 (0.8807-0.9964) |
| Stage5 >2y | LSTM | 24329 | 0.9497 (0.7732-0.9944) |
| Stage5 >2y | RSF | 24329 | 0.9441 (0.7650-0.9858) |
| Stage5 >2y | Qwen3-4B SFT | 24329 | 0.9715 (0.8752-0.9969) |
| Stage5 >2y | Qwen3-4B baseline | 24329 | 0.6455 (0.4120-0.8581) |
| Stage5 >2y | Llama3.1-8B | 24329 | 0.5781 (0.4799-0.6364) |
| Stage5 >2y | Huatuo-O1-7B | 24329 | 0.5683 (0.4606-0.6634) |

### Target 1y

| Stage | Model | N | AUROC (95% CI) |
|---|---|---:|---:|
| Stage1 0d-1m | XGBoost | 17126 | 0.7303 (0.6517-0.7993) |
| Stage1 0d-1m | LSTM | 17126 | 0.6609 (0.5507-0.7620) |
| Stage1 0d-1m | RSF | 17126 | 0.7318 (0.6643-0.7973) |
| Stage1 0d-1m | Qwen3-4B SFT | 17126 | 0.7432 (0.6712-0.8132) |
| Stage1 0d-1m | Qwen3-4B baseline | 17126 | 0.5704 (0.5219-0.6191) |
| Stage1 0d-1m | Llama3.1-8B | 17126 | 0.5120 (0.4837-0.5370) |
| Stage1 0d-1m | Huatuo-O1-7B | 17126 | 0.4913 (0.4637-0.5182) |
| Stage2 2m-3m | XGBoost | 8483 | 0.7979 (0.7005-0.8739) |
| Stage2 2m-3m | LSTM | 8483 | 0.7554 (0.6272-0.8596) |
| Stage2 2m-3m | RSF | 8483 | 0.7482 (0.6197-0.8461) |
| Stage2 2m-3m | Qwen3-4B SFT | 8483 | 0.7134 (0.6184-0.7964) |
| Stage2 2m-3m | Qwen3-4B baseline | 8483 | 0.5960 (0.5142-0.6682) |
| Stage2 2m-3m | Llama3.1-8B | 8483 | 0.5366 (0.5003-0.5712) |
| Stage2 2m-3m | Huatuo-O1-7B | 8483 | 0.5335 (0.4732-0.5938) |
| Stage3 4m-12m | XGBoost | 16788 | 0.7884 (0.7200-0.8780) |
| Stage3 4m-12m | LSTM | 16788 | 0.6457 (0.4572-0.7984) |
| Stage3 4m-12m | RSF | 16788 | 0.6874 (0.5482-0.7921) |
| Stage3 4m-12m | Qwen3-4B SFT | 16788 | 0.6368 (0.5245-0.7898) |
| Stage3 4m-12m | Qwen3-4B baseline | 16788 | 0.6111 (0.5420-0.6731) |
| Stage3 4m-12m | Llama3.1-8B | 16788 | 0.5414 (0.4982-0.5751) |
| Stage3 4m-12m | Huatuo-O1-7B | 16788 | 0.5543 (0.5115-0.5922) |
| Stage4 1y-2y | XGBoost | 11158 | 0.7163 (0.4869-0.9135) |
| Stage4 1y-2y | LSTM | 11158 | 0.6652 (0.4447-0.8249) |
| Stage4 1y-2y | RSF | 11158 | 0.6616 (0.4566-0.8384) |
| Stage4 1y-2y | Qwen3-4B SFT | 11158 | 0.6502 (0.5273-0.7976) |
| Stage4 1y-2y | Qwen3-4B baseline | 11158 | 0.6375 (0.5552-0.7122) |
| Stage4 1y-2y | Llama3.1-8B | 11158 | 0.5971 (0.5201-0.6607) |
| Stage4 1y-2y | Huatuo-O1-7B | 11158 | 0.5017 (0.4528-0.5492) |
| Stage5 >2y | XGBoost | 19992 | 0.9295 (0.7796-0.9919) |
| Stage5 >2y | LSTM | 19992 | 0.7272 (0.4530-0.8897) |
| Stage5 >2y | RSF | 19992 | 0.8008 (0.5931-0.9223) |
| Stage5 >2y | Qwen3-4B SFT | 19992 | 0.9214 (0.7972-0.9851) |
| Stage5 >2y | Qwen3-4B baseline | 19992 | 0.6195 (0.4535-0.7716) |
| Stage5 >2y | Llama3.1-8B | 19992 | 0.5684 (0.4757-0.6219) |
| Stage5 >2y | Huatuo-O1-7B | 19992 | 0.5469 (0.4522-0.6118) |

### Target 5y

| Stage | Model | N | AUROC (95% CI) |
|---|---|---:|---:|
| Stage1 0d-1m | XGBoost | 10295 | 0.6990 (0.6350-0.7619) |
| Stage1 0d-1m | LSTM | 10295 | 0.6742 (0.6019-0.7389) |
| Stage1 0d-1m | RSF | 10295 | 0.6780 (0.6129-0.7382) |
| Stage1 0d-1m | Qwen3-4B SFT | 10295 | 0.6994 (0.6295-0.7666) |
| Stage1 0d-1m | Qwen3-4B baseline | 10295 | 0.5553 (0.5162-0.5931) |
| Stage1 0d-1m | Llama3.1-8B | 10295 | 0.5328 (0.5082-0.5563) |
| Stage1 0d-1m | Huatuo-O1-7B | 10295 | 0.4997 (0.4772-0.5208) |
| Stage2 2m-3m | XGBoost | 4917 | 0.7917 (0.6817-0.8744) |
| Stage2 2m-3m | LSTM | 4917 | 0.7263 (0.6130-0.8181) |
| Stage2 2m-3m | RSF | 4917 | 0.6835 (0.5457-0.7948) |
| Stage2 2m-3m | Qwen3-4B SFT | 4917 | 0.6863 (0.5488-0.8056) |
| Stage2 2m-3m | Qwen3-4B baseline | 4917 | 0.5986 (0.5288-0.6505) |
| Stage2 2m-3m | Llama3.1-8B | 4917 | 0.5209 (0.4854-0.5525) |
| Stage2 2m-3m | Huatuo-O1-7B | 4917 | 0.5280 (0.4914-0.5572) |
| Stage3 4m-12m | XGBoost | 8922 | 0.7372 (0.5935-0.8537) |
| Stage3 4m-12m | LSTM | 8922 | 0.6399 (0.5256-0.7346) |
| Stage3 4m-12m | RSF | 8922 | 0.6344 (0.5185-0.7521) |
| Stage3 4m-12m | Qwen3-4B SFT | 8922 | 0.6439 (0.4962-0.7674) |
| Stage3 4m-12m | Qwen3-4B baseline | 8922 | 0.5909 (0.5180-0.6518) |
| Stage3 4m-12m | Llama3.1-8B | 8922 | 0.4955 (0.4463-0.5400) |
| Stage3 4m-12m | Huatuo-O1-7B | 8922 | 0.5178 (0.4825-0.5537) |
| Stage4 1y-2y | XGBoost | 4942 | 0.6865 (0.5258-0.8382) |
| Stage4 1y-2y | LSTM | 4942 | 0.5937 (0.3903-0.7699) |
| Stage4 1y-2y | RSF | 4942 | 0.6027 (0.4419-0.7616) |
| Stage4 1y-2y | Qwen3-4B SFT | 4942 | 0.6386 (0.4868-0.7821) |
| Stage4 1y-2y | Qwen3-4B baseline | 4942 | 0.6364 (0.5527-0.7127) |
| Stage4 1y-2y | Llama3.1-8B | 4942 | 0.5452 (0.4832-0.6123) |
| Stage4 1y-2y | Huatuo-O1-7B | 4942 | 0.5260 (0.4592-0.5929) |
| Stage5 >2y | XGBoost | 4538 | 0.8241 (0.4593-0.9516) |
| Stage5 >2y | LSTM | 4538 | 0.7720 (0.4152-0.9214) |
| Stage5 >2y | RSF | 4538 | 0.7654 (0.4648-0.8765) |
| Stage5 >2y | Qwen3-4B SFT | 4538 | 0.6761 (0.5434-0.9190) |
| Stage5 >2y | Qwen3-4B baseline | 4538 | 0.5306 (0.3821-0.6878) |
| Stage5 >2y | Llama3.1-8B | 4538 | 0.5513 (0.4858-0.6134) |
| Stage5 >2y | Huatuo-O1-7B | 4538 | 0.5295 (0.4957-0.5711) |

## Notes

- Primary AUROC 95% CI uses patient-level cluster bootstrap percentile intervals.
- DeLong 95% CI is exported as sensitivity columns in the CSV metric files.
- Standalone-valid results are exported separately for audit and should not be mixed with common-cohort pairwise difference analysis.
