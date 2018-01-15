# rnn_text_and_seq2seq
A repository for implementations of seq2seq and other text-specific uses of RNNs and their variants.

<b>load_process_data.py</b>: Load and process quotes data from three different sources. Convert text data into sequences of word IDs. 

<b>encoder_decoder.py</b>: A basic encoder-decoder setup in tensorflow using the raw_rnn function. Objective is to reconstruct the quotes.

<b>multilayer_encoder_decoder_with_attention.py:</b> As above, but with multiple layers and with Luong attention implemented inside the raw_rnn. 
