import torch
import torch.nn as nn
from torchtyping import TensorType

def generate(model,new_chars:int, context: TensorType[int], context_length:int, int_to_char: int) -> str:
    generator = torch.manual_seed(0)
    initial_state = generator.get_state()
    res = []
    # context = B X T
    for i in range(new_chars):
        if len(context.T) > context_length:
            context = context[:, -context_length:]
        prediction = model(context) # B,T, V
        last_time_step = prediction[:, -1, :] # B, V
        probs = nn.functional.softmax(last_time_step, dim=-1)
        next_char = torch.multinomial(probs, 1, generator=generator)
        generator.get_state(initial_state)
        context = torch.cat((context, next_char), dim=-1) # B, T -> B, T + 1
        int_to_char[next_char.item()]
        res.append(int_to_char[next_char.item()])
        
    return ''.join(res)
    