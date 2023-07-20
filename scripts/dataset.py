import fire
import time, glob, torch, json, numpy as np, sys, random, csv
from pathlib import Path
from toolformer.datasets import *
from toolformer.model import *
from toolformer.tokenizer import *
from torch.utils.data import DataLoader
from toolformer.filtering import *

def load_model(ckpt_dir: str, tokenizer, local_rank: int, world_size: int, lora: bool) -> Transformer:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    
    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=8, **params)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    return model
    
    
def main(model_path, tokenizer_path, lr=1e-5, epochs=10, lora=True, opt=torch.optim.Adam):
    # setup
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    
    # model and tokenizer
    start_time = time.time()
    tokenizer = Tokenizer(tokenizer_path)
    model = load_model(model_path, tokenizer, local_rank, world_size, lora=lora)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # data
    start_time = time.time()
    d = []
    with open('/home/libs/toolformer/data/dataset.csv', 'r') as file: 
        reader = csv.reader(file)
        for row in reader: d.append(row)
    ds = PromptDS(d)
    dl = DataLoader(ds, batch_size=8, num_workers=4)
    
    # filter dataset for finetuning
    data = build_finetune_dataset(dl, model, tokenizer, return_tokens=False)
    with open('/home/libs/toolformer/data/finetune_dataset.csv', 'w', newline='') as file: 
        writer = csv.writer(file)
        for d in data: writer.writerow(d)
    print(f'Toolformer dataset of length {len(data)} created in {(time.time() - start_time) // 60} minutes and {(time.time() - start_time) % 60:.2f} seconds')
    
if __name__ == "__main__":
    fire.Fire(main)