""" Example handler file. """

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import runpod
import torch

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
tokenizer = AutoTokenizer.from_pretrained("yigagilbert/salt_language_ID")
model = AutoModelForSeq2SeqLM.from_pretrained("yigagilbert/salt_language_ID")


def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    text = job_input.get('text')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Note that we convert the text input to lower case
    inputs = tokenizer(text.lower(), return_tensors="pt").to(device)
    output = model.to(device).generate(
        **inputs,
        max_new_tokens=5,
    )
    result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return {"language": result}


runpod.serverless.start({"handler": handler})
