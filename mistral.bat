set HF_HOME=C:\Users\choi1\Downloads\RecFormer
python mistral_ollama.py --data_path finetune_data/Office --device 0 --fp16 --no_prompt
python mistral_ollama.py --data_path finetune_data/Games --device 0 --fp16 --no_prompt
python mistral_ollama.py --data_path finetune_data/Pet --device 0 --fp16 --no_prompt
python mistral_ollama.py --data_path finetune_data/Scientific --device 0 --fp16
python mistral_ollama.py --data_path finetune_data/Instruments --device 0 --fp16
python mistral_ollama.py --data_path finetune_data/Arts --device 0 --fp16
python mistral_ollama.py --data_path finetune_data/Office --device 0 --fp16
python mistral_ollama.py --data_path finetune_data/Games --device 0 --fp16
python mistral_ollama.py --data_path finetune_data/Pet --device 0 --fp16
