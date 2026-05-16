# Experiment

## start with introducing new share token to both model as of new task indicator

## the model have two parts, 1. vision part is vision encoder 2. text part is decoder.

## first , imagine a light tint glass that is where vision encoder look through while trying to guess a object based on what it know.
## then , training vision encoder to understand shape via reconstructing same shape + hint (a label from example that use same embedding as text (or one that from vision-encoder))
## hint is much like a direct answer but in different language , which send to person next room to describe it (in this case , text decoder) 
## then, give or take packing reconstructed output into simple linear layer (hope, that it get some activation) with embedded hint (dimension is 2x now)
## then, pass it to text decoder (with already packed hint) and let decoder output text
## both output from encoder and decoder get project into same latent space which where will calculate loss (grpo)
## in training, the decoder must give out text ,preferably hint and since output from both have share same embedded value (hint) , then its must be easy to score by cluster group.
## if not give out text or give out hint then punishing decoder (might punish vision encoder in case it give wrong hint)
## in evaluation, after trials of error , the training wheel of this (hint) being take out , the only thing left is raw performance of decoder,
## then, giving score ,but this time only punish decoder that only rely on hint

## finally , output should be description text that get from vision extracted data and not on label hint


## Need
## vision encoder / text decoder this case is HuggingFaceTB/SmolVLM-Instruct ,since it's lightweight and have trained on Instruct (easy to focus target), which will be use its encoder for both text and vision and its decoder (text output often decoder only by standard, and encoder can be anything since will put embedded data in share latent anyways)
## dataset will be fault with label or really just anything with image+text
## vision encoder need to be masked for reconstructing.