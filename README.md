# rnn_text_and_seq2seq
A repository for implementations of seq2seq and other text-specific uses of RNNs and their variants.

load_process_data.py: Load and process quotes data from three different sources. Convert text data into sequences of word IDs. 

encoder_decoder.py: A basic encoder-decoder setup in tensorflow using the raw_rnn function. Objective is to reconstruct the quotes.

multilayer_encoder_decoder_with_attention.py: As above, but with multiple layers and with Luong attention implemented inside the raw_rnn. 
