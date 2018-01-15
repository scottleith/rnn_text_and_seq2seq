import tensorflow as tf

# Walking through the example at https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb
# but with actual text data we get from elsewhere.

enc_hidden_units = 1000
dec_hidden_units = enc_hidden_units*2 # because the encoder will be bidirectional.
embedding_size = 300

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Create placeholders for the encoder / decoder inputs, lengths, and targets.
enc_inp = tf.placeholder( shape = (None, None), dtype = tf.int32, name = "encoder_inputs" )
enc_inp_len = tf.placeholder( shape = (None,), dtype = tf.int32, name = "encoder_input_lengths" )
dec_inp = tf.placeholder( shape = (None, None), dtype = tf.int32, name = "decoder_inputs" )
#dec_inp_len = tf.placeholder( shape = (None,), dtype = tf.int32, name = "decoder_input_lengths" ) # Possibly unnecessary. 
dec_tgt = tf.placeholder( shape = (None, None), dtype = tf.int32, name = "decoder_targets") 

# For when embeddings are provided directly, instead of trained and looked up:
# enc_inp_emb = tf.placeholder( shape = (None, None), dtype = tf.float32, name = "encoder_input_embeddings")

# For when embeddings are trained and looked up.
# NOTE: In this case, enc_inp consists of sequences of word IDs. 
embeddings = tf.Variable( tf.random_uniform( [vocab_size, embedding_size], -1., 1.), dtype = tf.float32 )
enc_inp_emb = tf.nn.embedding_lookup( embeddings, enc_inp )
dec_inp_emb = tf.nn.embedding_lookup( embeddings, dec_inp )



# Encoder - Bidirectional RNN
enc_cell_fw = tf.contrib.rnn.LSTMCell( enc_hidden_units )
enc_cell_bw = tf.contrib.rnn.LSTMCell( enc_hidden_units )

( (enc_fw_output, enc_bw_output),(enc_fw_finalstate, enc_bw_finalstate) ) = tf.nn.bidirectional_dynamic_rnn( cell_fw = enc_cell_fw, 
									cell_bw = enc_cell_bw, 
									inputs = enc_inp_emb, 
									sequence_length = enc_inp_len,
									dtype = tf.float32, 
									time_major = True )

enc_outputs = tf.concat( (enc_fw_output, enc_bw_output), 2 ) # concatenate along third axis - the hidden state
enc_finalstate_c = tf.concat( (enc_fw_finalstate.c, enc_bw_finalstate.c), 1 ) # concat along second axis - cell state
enc_finalstate_h = tf.concat( (enc_fw_finalstate.h, enc_bw_finalstate.h), 1 ) # concat along second axis - hidden state
enc_finalstate = tf.contrib.rnn.LSTMStateTuple( c = enc_finalstate_c, h = enc_finalstate_h )


# Decoder - raw_rnn
# We are using tf.raw_rnn rather than the tf.nn.dynamic_rnn because the dynamic rnn does not allow us to feed
# decoder-produced tokens as input for the next timestep. 
dec_cell = tf.contrib.rnn.LSTMCell( dec_hidden_units )
enc_max_time, batch_size = tf.unstack( tf.shape( enc_inp ) ) # Assumes zero-padding has already occurred.
dec_len = enc_inp_len + 1 # Account for the <GO> tags that will be inserted at the start. 

# To get the output words at each decoder timestep, we must produce an actual word id prediction each time.
# So, we create our own global output variables to call at each timestep:
Wout = tf.Variable( tf.random_uniform( [dec_hidden_units, vocab_size], -1., 1.), dtype = tf.float32, name = "weights_output" )
bout = tf.Variable( tf.zeros( [vocab_size] ), dtype = tf.float32, name = "biases_output" )

eos_time_slice = tf.fill( [batch_size], 2, name = 'EOS')
go_time_slice = tf.ones( [batch_size], dtype = tf.int32, name = 'GO' )
pad_time_slice = tf.zeros( [batch_size], dtype = tf.int32, name = 'PAD')

eos_step_embedded = tf.nn.embedding_lookup( embeddings, eos_time_slice )
go_step_embedded = tf.nn.embedding_lookup( embeddings, go_time_slice ) # All IDs are 1s.
pad_step_embedded = tf.nn.embedding_lookup( embeddings, pad_time_slice ) # All IDs are 0s. 


# The raw_rnn requires its initial state and transition behaviour to be defined.
# Initial state:

def loop_fn_initial():
	initial_elements_finished = (0 >= dec_len) # All False at first timestep.
	initial_inp = go_step_embedded # Provide the <EOS> embedding as first x_t input.
	initial_cell_state = enc_finalstate # Provide final state of encoder as initial decoder state.
	initial_cell_output = None # No output yet.
	initial_loop_state = None # Don't have any other information to pass.
	return( 
		initial_elements_finished, 
		initial_inp, 
		initial_cell_state, 
		initial_cell_output, 
		initial_loop_state
		)

# Transition behaviour:
def loop_fn_transition( time, prev_output, prev_state, prev_loop_state ):
	def get_next_input():
		output_logits = tf.add( tf.matmul( prev_output, Wout) , bout)
		pred = tf.argmax( output_logits, axis = 1 )  
		next_input = tf.nn.embedding_lookup( embeddings, pred ) 
		return next_input # The embedding for the word produced this timestep. 
	elements_finished = (time >= dec_len) # Produces a boolnea tensor of [batch_size] which defines if the sequence has ended
	finished = tf.reduce_all( elements_finished ) # Boolean scalar, False as long as there is one False
	# i.e., are we finished? Unless all time > current dec_len (e.g., the corresponding enc_len+5),
	# we continue. 
	input_next = tf.cond( finished, lambda: pad_step_embedded, lambda: get_next_input() ) # If finished = True, 
	# this returns pad_step_embedded (sequence is over), otherwise returns get_next_input() and 
	# continues the loop. 
	state = prev_state # beginning state for next timestep
	output = prev_output # beginning output for next timestep
	loop_state = None # As above, no other information to pass. 
	return( elements_finished, input_next, state, output, loop_state )

# Combine the initialization and transition functions into single call that will check 
# if the state is None and return init or transition.
def loop_fn( time, prev_output, prev_state, prev_loop_state ):
	if prev_state is None:
		assert prev_output is None and prev_state is None
		return loop_fn_initial()
	else:
		return loop_fn_transition( time, prev_output, prev_state, prev_loop_state )


dec_outputs_ta, dec_finalstate, _ = tf.nn.raw_rnn( dec_cell, loop_fn )

dec_outputs = dec_outputs_ta.stack() 

# tf.unstack will take the provided tensor and divide it along the values of the given axis (base axis = 0).
# So if something has shape (A,B,C,D) and we just call tf.unstack, output is a tensor of A tensors of shape (B,C,D).
# This is supposed to take apart a [ time, batch, hidden_units ] shape tensor
dec_max_steps, dec_batch_size, dec_dim = tf.unstack( tf.shape(dec_outputs) )
enc_max_steps, enc_batch_size = tf.unstack( tf.shape(dec_tgt) )
dec_outputs_flat = tf.reshape( dec_outputs, [-1, dec_hidden_units] ) # Flattens to [ t * b, h ] from [ t, b, h ]. 
dec_logits_flat = tf.add( tf.matmul( dec_outputs_flat, Wout ), bout )
dec_logits = tf.reshape( dec_logits_flat, (dec_max_steps, dec_batch_size, vocab_size) )
dec_predict = tf.argmax( dec_logits, 2 ) # argmax over 3rd dimension (vocab_size) - which word is most likely?
dec_labels = tf.one_hot( dec_tgt, depth = vocab_size, dtype = tf.float32 ) # One-hot our labels and we're ready.
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels = dec_labels, logits = dec_logits  )

loss = tf.reduce_mean( stepwise_cross_entropy )
train_op = tf.train.AdamOptimizer().minimize( loss )
loss_track = []

# Train our model!

sess.run( tf.global_variables_initializer())

def run_epoch():
	encoder_inputs, encoder_len = generate_batches( inputs, 100 )
	decoder_inputs = [ np.insert( i, 0, np.ones((i.shape[1])), axis = 0 ) for i in encoder_inputs ]
	decoder_targets = [ np.append( i, np.full((1,i.shape[1]), fill_value = 2 ), axis = 0 ) for i in encoder_inputs ]
	combined = list( zip( encoder_inputs, encoder_len, decoder_inputs, decoder_targets ) )
	shuffle(combined)
	encoder_inputs[:], encoder_len[:], decoder_inputs[:], decoder_targets[:] = zip(*combined)
	try:
		for i in range( len(encoder_inputs) ):
			fd = { 
				enc_inp: encoder_inputs[i], 
				enc_inp_len: encoder_len[i], 
				dec_inp: decoder_inputs[i],
				dec_tgt: decoder_targets[i] 
			}
			_, l = sess.run( [train_op, loss], feed_dict = fd )
			loss_track.append(l)
			if i == 0 or i % 10 == 0:
				print( 'batch {}'.format(i) )
				print( '  minibatch training loss: {}'.format( sess.run(loss,fd) ) )
				predict_ = sess.run( dec_predict, fd )
				for j, (inp, pred) in enumerate( zip( fd[enc_inp].T, predict_.T) ):
					inp = [ int2word[i] for i in inp ]
					pred = [ int2word[i] for i in pred ]
					print( '  sample {}:'.format(j+1) )
					print( '  input  > {}'.format(inp) )
					print( '  predicted > {}'.format(pred) )
					if j > 4:
						break


for i in range(60):
	run_epoch()


