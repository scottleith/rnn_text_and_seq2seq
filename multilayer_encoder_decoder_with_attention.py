from tensorflow.contrib.rnn import MultiRNNCell, BasicLSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import dynamic_decode, BasicDecoder, TrainingHelper
from tensorflow.python.ops import math_ops, array_ops

# Multilayer encoder-decoder with Luong attention.
# NOTE: The extra layer(s) and added attention don't actually help in this context.
# //TODO: Get some better, larger-scale data from NLTK. Test on longer sequences. 

num_layers = 1
enc_hidden_units = 500
dec_hidden_units = enc_hidden_units*2
embed_size = 300
enc_inp = tf.placeholder( shape = (None, None), dtype = tf.int32, name = "encoder_inputs" )
enc_inp_len = tf.placeholder( shape = (None,), dtype = tf.int32, name = "encoder_input_lengths" )
dec_inp = tf.placeholder( shape = (None, None), dtype = tf.int32, name = "decoder_inputs" )
dec_tgt = tf.placeholder( shape = (None, None), dtype = tf.int32, name = "decoder_targets") 

embeddings = tf.Variable( tf.random_uniform( [vocab_size, embed_size], -1., 1.), dtype = tf.float32 )
enc_inp_emb = tf.nn.embedding_lookup( embeddings, enc_inp )
dec_inp_emb = tf.nn.embedding_lookup( embeddings, dec_inp )

enc_max_time, batch_size = tf.unstack( tf.shape( enc_inp ) ) # Assumes zero-padding has already occurred.
dec_len = enc_inp_len + 1 

# Encoder - Bidirectional RNN

if num_layers > 1:
	cell_fw = MultiRNNCell( [ BasicLSTMCell( enc_hidden_units ) for _ in range(num_layers) ] )
	cell_bw = MultiRNNCell( [ BasicLSTMCell( enc_hidden_units ) for _ in range(num_layers) ] )
	dec_cell = MultiRNNCell( [ BasicLSTMCell( dec_hidden_units ) for _ in range(num_layers) ] )
elif num_layers == 1:
	cell_fw, cell_bw = BasicLSTMCell( enc_hidden_units ), BasicLSTMCell( enc_hidden_units )
	dec_cell = BasicLSTMCell( dec_hidden_units )

( (enc_fw_output, enc_bw_output),(enc_fw_finalstate, enc_bw_finalstate) ) = tf.nn.bidirectional_dynamic_rnn( cell_fw = cell_fw, 
									cell_bw = cell_bw, 
									inputs = enc_inp_emb, 
									sequence_length = enc_inp_len,
									dtype = tf.float32, 
									time_major = True )

enc_outputs = tf.concat( (enc_fw_output, enc_bw_output),-1 )

if num_layers == 1:
	enc_finalstate_c = tf.concat( (enc_fw_finalstate.c, enc_bw_finalstate.c), 1 ) # concat along second axis - cell state
	enc_finalstate_h = tf.concat( (enc_fw_finalstate.h, enc_bw_finalstate.h), 1 ) # concat along second axis - hidden state
	enc_finalstate = tf.contrib.rnn.LSTMStateTuple( c = enc_finalstate_c, h = enc_finalstate_h )
elif num_layers > 1:
	enc_finalstate = []
	for i in range(num_layers):
		enc_state_c = tf.concat( (enc_fw_finalstate[i].c, enc_bw_finalstate[i].c), 1, name = "bidirect_concat_c" )
		enc_state_h = tf.concat( (enc_fw_finalstate[i].h, enc_bw_finalstate[i].h), 1, name = "bidirect_concat_h" )
		encoder_state = LSTMStateTuple(c=enc_state_c, h=enc_state_h)
		enc_finalstate.append(encoder_state)
		enc_finalstate = tuple( enc_finalstate )

enc_max_time, batch_size = tf.unstack( tf.shape( enc_inp ) ) # Assumes zero-padding has already occurred.
dec_inp_len = enc_inp_len + 1

w = {
	'align': tf.Variable( tf.random_uniform( [dec_hidden_units, dec_hidden_units], -.01, .01), dtype = tf.float32, name = 'weights_align' ),
	'context': tf.Variable( tf.random_uniform( [dec_hidden_units*2, dec_hidden_units*2], -.01, .01), dtype = tf.float32, name = 'weights_context' ),
	'dec_out': tf.Variable( tf.random_uniform( [dec_hidden_units*2, vocab_size], -.01, .01), dtype = tf.float32, name = 'weights_output' )
}
b = {
	'context': tf.Variable( tf.zeros([dec_hidden_units*2,]), dtype = tf.float32, name = 'bias_context' ),
	'dec_out': tf.Variable( tf.zeros([vocab_size,]), dtype = tf.float32, name = 'bias_output' )
}


# We are using tf.raw_rnn rather than the tf.nn.dynamic_rnn because the dynamic rnn does not allow us to feed
# decoder-produced tokens as input for the next timestep. 

eos_time_slice = tf.fill( [batch_size], 2, name = 'EOS')
go_time_slice = tf.ones( [batch_size], dtype = tf.int32, name = 'GO' )
pad_time_slice = tf.zeros( [batch_size], dtype = tf.int32, name = 'PAD')

eos_step_embedded = tf.nn.embedding_lookup( embeddings, eos_time_slice )
go_step_embedded = tf.nn.embedding_lookup( embeddings, go_time_slice ) # All IDs are 1s.
pad_step_embedded = tf.nn.embedding_lookup( embeddings, pad_time_slice ) # All IDs are 0s. 


# Initial state:

def loop_fn_initial():
	initial_elements_finished = (0 >= dec_inp_len) # All False at first timestep.
	initial_inp = eos_step_embedded # Provide the <EOS> embedding as first x_t input.
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
		# NOTE: prev_output is [ batch, hidden ]
		enc_out = tf.transpose( enc_outputs, [1,2,0] ) # Transpose to [ batch, hidden, time ] 
		prev_out = tf.matmul( prev_output, w['align'] )# Outputs [ batch, hidden ]
		prev_out = tf.expand_dims( prev_out, 1 ) # Expand size to [ batch, 1, hidden ]
		attn_wts = tf.nn.softmax( tf.matmul( prev_out, enc_out) ) # Outputs [ batch, 1, time ]
		enc_out = tf.transpose( enc_outputs, [1,0,2]) # Output [ batch, time, hidden ] 
		context = tf.matmul( attn_wts, enc_out ) # Outputs [ batch, 1, hidden ]
		context = tf.squeeze( context, [1] ) # Eliminates the 1, outputs [ batch, hidden ]
		context = tf.concat( (context, prev_output), axis = 1 ) # Outputs [ batch, hidden*2 ]
		out = tf.nn.tanh( tf.add( tf.matmul(context,w['context']), b['context'] ) ) # Outputs [ batch, hidden*2 ]
		output_logits = tf.add( tf.matmul(out, w['dec_out']) , b['dec_out'] ) # Outputs [ batch, vocabulary_size ]
		pred = tf.argmax( output_logits, axis = 1 ) # Get max (i.e., word ID) along vocabulary axis. 
		next_input = tf.nn.embedding_lookup( embeddings, pred ) # Look up the embedding, and we have our next input!
		return next_input
	elements_finished = (time >= dec_inp_len)
	finished = tf.reduce_all( elements_finished )
	input_next = tf.cond( finished, lambda: pad_step_embedded, lambda: get_next_input() )
	state = prev_state
	output = prev_output
	loop_state = None
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
dec_max_steps, dec_batch_size, dec_dim = tf.unstack( tf.shape(dec_outputs) )



# NOTE FOR THIS SECTION: Recall that in tf.matmul, [a,b,c] * [a,c,d] = [a,b,d].

enc_out_temp = tf.transpose( enc_outputs, [1,2,0] ) # Transpose to [ batch, hidden, time ]
dec_out_temp = tf.transpose( dec_outputs, [1,0,2]) # Transpose to [ batch, time, hidden ] 
attn_wts = tf.nn.softmax( tf.matmul( dec_out_temp, enc_out_temp), dim = 1 ) # Outputs [ batch, time, time ]

enc_out_temp = tf.transpose( enc_outputs, [1,0,2] ) # Transpose to [ batch, time, hidden]
context = tf.matmul( attn_wts, enc_out_temp ) # Outputs [ batch, time, hidden ]
dec_output = tf.concat( (context, dec_out_temp), axis = 2 ) # Outputs [ batch, time, hidden*2 ]
dec_output_flat = tf.reshape( dec_output, [-1, dec_hidden_units*2] ) # [ batch*time, hidden*2 ]

# Recall that w['context'] is [ hidden*2, hidden*2 ].
# The below outputs [ batch*time, hidden*2 ]
out = tf.nn.tanh( tf.add( tf.matmul(dec_output_flat,w['context']), b['context'] ) )

output_logits = tf.add( tf.matmul(out, w['dec_out']) , b['dec_out'] ) # Outputs [ b*t, v ]
output_logits = tf.reshape( output_logits, (dec_max_steps, dec_batch_size, vocab_size)) # [ t, b, v ]. We've arrived!

dec_predict = tf.argmax( output_logits, axis = 2 ) # Get our maxes along the vocabulary axis (i.e., each word ID) 
dec_labels = tf.one_hot( dec_tgt, depth = vocab_size, dtype = tf.float32) # Transform our IDs into one-hots.

# We're ready to compute the cost!
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels = dec_labels, logits = output_logits  )

loss = tf.reduce_mean( stepwise_cross_entropy )
train_op = tf.train.AdamOptimizer(learning_rate = .001).minimize( loss )
loss_track = []

sess = tf.InteractiveSession()
sess.run( tf.global_variables_initializer())

def run_epoch():
	encoder_inputs, encoder_len = generate_batches( inputs, 50 )
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
	except KeyboardInterrupt:
		print( 'Training Interrupted.' )


for i in range(60):
	run_epoch()
	if KeyboardInterrupt:
		break
