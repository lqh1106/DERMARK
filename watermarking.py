import torch
from hashlib import sha256
from bit_distributing import fitpartition
import numpy as np
from robustness import dynamic_programming_segment
def default_hash_fn(tensor):
    """Returns the hash of the given tensor using the sha256 algorithm.

    Args:
        tensor: The tensor to hash.

    Returns:
        int: The hash of the tensor.
    """
    return int(sha256(str(tensor).encode('utf-8')).hexdigest(), 16) % (10 ** 8)


@torch.no_grad()
def generate(model,vocab_size, prior_tokens, mark_info, alpha, max_length ,delta,hash_function=default_hash_fn):
    with torch.no_grad():
        B, T = prior_tokens.shape
        device = prior_tokens.device
        l_t = torch.zeros((max_length, B, vocab_size))
        par = fitpartition(alpha, delta, mark_info, B)
        generated_tokens = prior_tokens.clone()
        in_green_list = torch.zeros(B, dtype=int).to(device)
        bit_count = 0
        N = [0]*B
        
        for _ in range(max_length):
            
            l_t_batch = model(generated_tokens)[:, -1, :]
            seeds = [hash_function(generated_tokens[i, -1]) for i in range(B)]
            generators = [torch.Generator(
                device=device).manual_seed(seed) for seed in seeds]

            vs = l_t_batch.shape[-1]  # Vocabulary size
            gls = int(0.5 * vs)  # Green list size
            gli_batch = torch.stack(
                [torch.randperm(vs, generator=generators[i], device=device) for i in range(B)])
            l_t_input_batch = l_t_batch
            
            bit_batch,N,bit_count,sig = par.markbit_by_partition(
                l_t_input_batch, gli_batch, gls, in_green_list,_)
            gain = [(gli_batch[i] < gls) if bit_batch[i] > 0.5 else (gli_batch[i] >= gls) for i in range(B)]
            gain = [delta * g for g in gain]
                # Apply watermarking
            gain = torch.stack(gain, dim=0)
            l_t_batch = l_t_batch + gain
            # Apply softmax and sample the next token
            l_t_batch = torch.softmax(l_t_batch, dim=-1)
            
            next_tokens = torch.multinomial(l_t_batch, 1)
            
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=-1)
            if B == 1:
                print("(",_,end=",")
                if gli_batch[0][next_tokens[0].item()] > gls:
                # if (gli_batch.gather(1,generated_tokens[:, -1].unsqueeze(-1))).squeeze():
                    print(f"\033[92m{next_tokens[0].item()}\033[0m", end=",")
                else:
                    print(f"\033[91m{next_tokens[0].item()}\033[0m", end=",") 
            print(")",end=",")
            in_green_list += (gli_batch.gather(1,
                        generated_tokens[:, -1].unsqueeze(-1)) < gls).squeeze()
        torch.cuda.empty_cache()
        l_t = l_t.to(model.device)
        return generated_tokens,N,bit_count,l_t

@torch.no_grad()
def detect_robustness(model, prior, ids, vocab_size, alpha, delta,N_batch, bit_count,hash_function=default_hash_fn):
    with torch.no_grad():
        B, T = ids.shape
        device = ids.device
        gls = int(0.5 * vocab_size)  # Green list size
        N = [0 for _ in range(B)]
        bit = [[]]*B
        P =[[]]*B
        bit = np.array(bit)
        P = np.array(P)
        in_green_list = torch.zeros(B, dtype=int).to(device)
        # par = exapartition(alpha, delta, [], B)
        for i in range(prior.shape[1] - 1, T - 1):
            # Seed generators and calculate green list
            
            seeds = [hash_function(ids[j, i]) for j in range(B)]
            generators = [torch.Generator(
                device=device).manual_seed(seed) for seed in seeds]
            gli_batch = torch.stack([torch.randperm(
                vocab_size, generator=generators[i], device=device) for i in range(B)])
            # Count green list occurrences
            
            l_t_batch = model(ids[:, :i + 1])[:, -1, :]
            
            in_green_list += (gli_batch.gather(1,
                                ids[:, i + 1].unsqueeze(-1)) < gls).squeeze()
                        
            current_bit = (gli_batch.gather(1,ids[:, i + 1].unsqueeze(-1)) < gls).squeeze()
            N = [n+1 for n in N]
            if B > 1:
                current_bit = [[int(i)] for i in current_bit]
            else:
                current_bit = [[int(current_bit.item())]]
            current_bit = np.array(current_bit)
            bit = np.hstack((bit, current_bit))

            
            current_p = []
            for l_t,gli in zip(l_t_batch,gli_batch):
                l_t = torch.softmax(l_t, dim=-1)
                z_1 = l_t * (gli < gls)
                sigma_1 = torch.sum(z_1).item()
                current_p.append([sigma_1])
            current_p = np.array(current_p)
            P = np.hstack((P, current_p))
            
        beta = 0.3
        info = []
        for j,N,count in zip(range(B),N_batch,bit_count):
            if N == 0:
                b,beta = dynamic_programming_segment(P[j], bit[j],alpha,count,delta,beta)
            else:
                b,beta = dynamic_programming_segment(P[j][:-N], bit[j][:-N],alpha,count,delta,beta)
            info.append(b)
        return info
