# -*- coding: utf-8 -*-
import codecs
import nltk
from nltk.stem import SnowballStemmer
import theano.tensor as T
import cPickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from common_functions import Conv_with_Mask

def load_word2vec_2M():
	word2vec = {}
	print "==> loading 300d word2vec 2M"
	f = open('/project/mgh/Data/Jimmy/word2vec/crawl-300d-2M.vec', 'r')
	for line in f:
		l = line.split()
		word2vec[l[0]] = map(float, l[1:])

	print "==> word2vec is loaded"
	return word2vec

def create_asymmetric_dicts(u, v, tokenizer,threshold, model, u_dict_name, v_dict_name):
	u_vectors = model.get_weights()[0][0]
	v_vectors = model.get_weights()[1][0]

	for word, i in tokenizer.word_index.items():
		u[word] = list(u_vectors[i])
		v[word] = list(v_vectors[i])
        
	with open(u_dict_name, 'wb') as handle:
		 cPickle.dump(u, handle)

	with open(v_dict_name, 'wb') as handle:
		cPickle.dump(v, handle)
	return u, v

def load_wordpair_to_init_l(rand_values, ivocab, word2vec):
	fail = 0
	with open('u_UNK_0829_' + str(threshold) + '_.pickle', 'rb') as handle:
		l1_word2vec = cPickle.load(handle)
	for id, word in ivocab.iteritems():
		emb = None
		if l1_word2vec.get(word.lower()) != None:
			emb = l1_word2vec.get(word.lower())
		else:
			emb = l1_word2vec.get('unk2')
		if emb is not None:
			rand_values[id] = np.array(emb)
		else:
			fail += 1
	print '==> use wordpair initialization over...fail ', fail
	return rand_values

def load_wordpair_to_init_r(rand_values, ivocab, word2vec):
	fail = 0
	with open('v_UNK_0829_' + str(threshold) + '_.pickle', 'rb') as handle:
		r1_word2vec = cPickle.load(handle)
	for id, word in ivocab.iteritems():
		emb = None
		if r1_word2vec.get(word.lower()) != None:
			emb = r1_word2vec.get(word.lower())
		else:
			emb = r1_word2vec.get('unk1')
		if emb is not None:
			rand_values[id] = np.array(emb)
		else:
			fail += 1
	print '==> use wordpair initialization over...fail ', fail
	return rand_values


def write_word_pair_to_txt(word_pair, dir_path, file_name):
	with open(dir_path + file_name, 'wb') as f:
		for x, y in word_pair:
			f.write('(%s,%s)\n' %(x, y))
			f.write('(%s,%s)\n' %(x, 'unk1'))
			f.write('(%s,%s)\n' %('unk2', y))

def clean_word_pair(word_pair, non_entail_word_pair):
	for pair in non_entail_word_pair:
		if pair in word_pair:
			word_pair.remove(pair)
	return word_pair

def create_neg_wordpairs(word2vec, non_entail_threshold):
	root = "/nethome/cwu392/NLP/SciTailV1.1/tsv_format/"
	file_name = 'scitail_1.0_train.tsv'
	unseen = []
	non_entail_word_pair = set()
	readfile = codecs.open(root + file_name, 'r', 'utf-8')
	stemmer = SnowballStemmer("english")
	for line in readfile:
			parts = line.strip().split('\t')
			if len(parts) == 3:
				# label = 'neural' or 'entails'
				label = parts[2]
				if label != 'entails':
					#save premise in sentence_wordlist_l (l:left)
					sentence_wordlist_l = [x for x in nltk.word_tokenize(parts[0].strip()) if x.isalpha()]
					#save hypothesis in sentence_wordlist_l (l:left)
					sentence_wordlist_r = [x for x in nltk.word_tokenize(parts[1].strip()) if x.isalpha()]
					#conver to lower case
					sentence_wordlist_l = [x.lower() for x in sentence_wordlist_l] 
					sentence_wordlist_r = [x.lower() for x in sentence_wordlist_r]
					#create a matrix to find max cosine similarity
					row, col = len(sentence_wordlist_l), len(sentence_wordlist_r)
					cos_matrix = [[0 for _ in range(col)] for _ in range(row)]
					for r in range(row):
						for c in range(col):
							# If words in premise are unseen in word vector dict, save in unseen
							if sentence_wordlist_l[r] not in word2vec:
								unseen.append(sentence_wordlist_l[r])
							# If words in hypothesis are unseen in word vector dict, save in unseen
							elif sentence_wordlist_r[c] not in word2vec:
								unseen.append(sentence_wordlist_r[c])
							# Find word pairs
							elif sentence_wordlist_l[r] in word2vec and sentence_wordlist_r[c] in word2vec:
								v1 = word2vec.get(sentence_wordlist_l[r])
								v2 = word2vec.get(sentence_wordlist_r[c])
								cos_matrix[r][c] = float(cosine_similarity([v1], [v2]))
						if max(cos_matrix[r]) >= non_entail_threshold:
							idx = cos_matrix[r].index(max(cos_matrix[r]))
							if sentence_wordlist_l[r] != sentence_wordlist_r[idx] and \
								stemmer.stem(sentence_wordlist_l[r]) != stemmer.stem(sentence_wordlist_r[idx]):
								non_entail_word_pair.add((sentence_wordlist_l[r], sentence_wordlist_r[idx]))
	return non_entail_word_pair

def create_pos_wordpairs(word2vec, threshold):
	root = "/nethome/cwu392/NLP/SciTailV1.1/tsv_format/"
	file_name = 'scitail_1.0_train.tsv'
	unseen = []
	word_pair = set()
	readfile = codecs.open(root + file_name, 'r', 'utf-8')
	for line in readfile:
			parts = line.strip().split('\t')
			if len(parts) == 3:
				# label = 'neural' or 'entails'
				label = parts[2]
				if label == 'entails':
					#save premise in sentence_wordlist_l (l:left)
					sentence_wordlist_l = [x for x in nltk.word_tokenize(parts[0].strip()) if x.isalpha()]
					#save hypothesis in sentence_wordlist_l (l:left)
					sentence_wordlist_r = [x for x in nltk.word_tokenize(parts[1].strip()) if x.isalpha()]
					#conver to lower case
					sentence_wordlist_l = [x.lower() for x in sentence_wordlist_l] 
					sentence_wordlist_r = [x.lower() for x in sentence_wordlist_r]
					#create a matrix to find max cosine similarity
					row, col = len(sentence_wordlist_l), len(sentence_wordlist_r)
					cos_matrix = [[0 for _ in range(col)] for _ in range(row)]
					for r in range(row):
						for c in range(col):
							# If words in premise are unseen in word vector dict, save in unseen
							if sentence_wordlist_l[r] not in word2vec:
								unseen.append(sentence_wordlist_l[r])
							# If words in hypothesis are unseen in word vector dict, save in unseen
							elif sentence_wordlist_r[c] not in word2vec:
								unseen.append(sentence_wordlist_r[c])
							# Find word pairs
							elif sentence_wordlist_l[r] in word2vec and sentence_wordlist_r[c] in word2vec:
								v1 = word2vec.get(sentence_wordlist_l[r])
								v2 = word2vec.get(sentence_wordlist_r[c])
								cos_matrix[r][c] = float(cosine_similarity([v1], [v2]))
						if max(cos_matrix[r]) >= threshold:
							idx = cos_matrix[r].index(max(cos_matrix[r]))
							word_pair.add((sentence_wordlist_l[r], sentence_wordlist_r[idx]))
	return word_pair

def selu(x):
	"""Compute the element-wise Scaled Exponential Linear unit [3]_.
	.. versionadded:: 0.9.0
	Parameters
	----------
	x : symbolic tensor
		Tensor to compute the activation function for.
	Returns
	-------
	symbolic tensor
		Element-wise scaled exponential linear activation function applied to `x`.
	References
	----------
	.. [3] Klambauer G, Unterthiner T, Mayr A, Hochreiter S.
		"Self-Normalizing Neural Networks" <https://arxiv.org/abs/1706.02515>
	"""
	alpha = 1.6732632423543772848170429916717
	scale = 1.0507009873554804934193349852946
	return scale * T.nnet.elu(x, alpha)

def load_word2vec_to_init_l(rand_values, ivocab, word2vec):
	fail = 0
	for id, word in ivocab.iteritems():
		emb = None
		if word2vec.get(word) != None:
			emb = word2vec.get(word)
		if emb is not None:
			rand_values[id] = np.array(emb)
		else:
			fail += 1
	print '==> use word2vec initialization over...fail ', fail
	return rand_values

def load_word2vec_to_init_r(rand_values, ivocab, word2vec):
	fail = 0
	for id, word in ivocab.iteritems():
		if word2vec.get(word) != None:
			emb = word2vec.get(word)
		if emb is not None:
			rand_values[id] = np.array(emb)
		else:
			fail += 1
	print '==> use word2vec initialization over...fail ', fail
	return rand_values

class Conv_for_Pair(object):
    """we define CNN by input tensor3 and output tensor3, like RNN, filter width must by 3,5,7..."""

    def __init__(self, rng, 
                 origin_input_tensor3, origin_input_tensor3_r, 
                 input_tensor3, input_tensor3_r, 
                 #our method
                 cm_origin_input_tensor3, cm_origin_input_tensor3_r, 
                 cm_input_tensor3, cm_input_tensor3_r,                  
                 ###########
                 mask_matrix, mask_matrix_r,
                 filter_shape, filter_shape_context, 
                 image_shape, image_shape_r, 
                 W, b, W_posi, b_posi, W_context, b_context, posi_emb_matrix, posi_emb_size, K_ratio):
        batch_size = origin_input_tensor3.shape[0]
        hidden_size = origin_input_tensor3.shape[1]

        l_len = origin_input_tensor3.shape[2]
        r_len = origin_input_tensor3_r.shape[2]
        #construct interaction matrix
        input_tensor3 = input_tensor3*mask_matrix.dimshuffle(0,'x',1)
        input_tensor3_r = input_tensor3_r*mask_matrix_r.dimshuffle(0,'x',1) #(batch, hidden, r_len)
        dot_tensor3 = T.batched_dot(input_tensor3.dimshuffle(0,2,1),input_tensor3_r) #(batch, l_len, r_len)
        # our method
        cm_input_tensor3 = cm_input_tensor3 * mask_matrix.dimshuffle(0,'x',1)
        cm_input_tensor3_r = cm_input_tensor3_r * mask_matrix_r.dimshuffle(0,'x',1) #(batch, hidden, r_len)
        cm_dot_tensor3 = T.batched_dot(cm_input_tensor3.dimshuffle(0,2,1), cm_input_tensor3_r) #(batch, l_len, r_len)
        
        new_dot_tensor3 = dot_tensor3 + 0.01 * cm_dot_tensor3
    
        '''
        try to get position shift of best match
        '''
        aligned_posi_l = T.argmax(dot_tensor3, axis=2).flatten()
        posi_emb_tensor3_l = posi_emb_matrix[aligned_posi_l].reshape((batch_size,dot_tensor3.shape[1],posi_emb_size)).dimshuffle(0,2,1) #(batch, emb_size, l_len)
        
        aligned_posi_r = T.argmax(dot_tensor3, axis=1).flatten()
        posi_emb_tensor3_r = posi_emb_matrix[aligned_posi_r].reshape((batch_size,dot_tensor3.shape[2],posi_emb_size)).dimshuffle(0,2,1) #(batch, emb_size, r_len)

        l_max_cos = (1.0 - T.min(selu(cm_dot_tensor3), axis=2))/(1.0 + 0.5 * T.max(selu(dot_tensor3), axis=2))#1.0/T.exp(T.max(T.nnet.sigmoid(dot_tensor3), axis=2)) #(batch, l_len)
        r_max_cos = (1.0 - T.min(selu(cm_dot_tensor3), axis=1))/(1.0 + 0.5 * T.max(selu(dot_tensor3), axis=1))#1.0/T.exp(T.max(T.nnet.sigmoid(dot_tensor3), axis=1)) #(batch, r_len)
        '''
        another interaction matrix
        '''

        dot_matrix_for_right = T.nnet.softmax(new_dot_tensor3.reshape((batch_size*l_len, r_len)))  #(batch*l_len, r_len)
        dot_tensor3_for_right = dot_matrix_for_right.reshape((batch_size, l_len, r_len))#(batch, l_len, r_len)
        weighted_sum_r = T.batched_dot(dot_tensor3_for_right, input_tensor3_r.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix.dimshuffle(0,'x',1) #(batch,hidden, l_len)

        dot_matrix_for_left = T.nnet.softmax(new_dot_tensor3.dimshuffle(0,2,1).reshape((batch_size*r_len, l_len))) #(batch*r_len, l_len)
        dot_tensor3_for_left = dot_matrix_for_left.reshape((batch_size, r_len, l_len))#(batch, r_len, l_len)
        
        weighted_sum_l = T.batched_dot(dot_tensor3_for_left, input_tensor3.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix_r.dimshuffle(0,'x',1) #(batch,hidden, r_len)

        #convolve left, weighted sum r
        biased_conv_model_l = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3*l_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_l = biased_conv_model_l.naked_conv_out
        self.conv_out_l = biased_conv_model_l.masked_conv_out
        self.maxpool_vec_l = biased_conv_model_l.maxpool_vec
        conv_model_l = Conv_with_Mask(rng, input_tensor3=T.concatenate([origin_input_tensor3,posi_emb_tensor3_l],axis=1),
                 mask_matrix = mask_matrix,
                 image_shape=(image_shape[0], image_shape[1], image_shape[2]+posi_emb_size, image_shape[3]),
                 filter_shape=(filter_shape[0],filter_shape[1],filter_shape[2]+posi_emb_size,filter_shape[3]), W=W_posi, b=b_posi)
        temp_conv_output_l = conv_model_l.naked_conv_out
        conv_model_weighted_r = Conv_with_Mask(rng, input_tensor3=weighted_sum_r,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_r = conv_model_weighted_r.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_l=T.repeat(mask_matrix.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_l=(1.0-mask_for_conv_output_l)*(mask_for_conv_output_l-10)

        self.biased_conv_attend_out_l = T.tanh(biased_temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        self.biased_attentive_sumpool_vec_l=T.sum(self.biased_conv_attend_out_l, axis=2)
        self.biased_attentive_meanpool_vec_l=self.biased_attentive_sumpool_vec_l/T.sum(mask_matrix,axis=1).dimshuffle(0,'x')
        masked_biased_conv_output_l=self.biased_conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_l=T.max(masked_biased_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        self.conv_attend_out_l = T.tanh(temp_conv_output_l+ temp_conv_output_weighted_r+ b_posi.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        masked_conv_output_l=self.conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_l=T.max(masked_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        #convolve right, weighted sum l
        biased_conv_model_r = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3_r*r_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_r = biased_conv_model_r.naked_conv_out
        self.conv_out_r = biased_conv_model_r.masked_conv_out
        self.maxpool_vec_r = biased_conv_model_r.maxpool_vec
        conv_model_r = Conv_with_Mask(rng, input_tensor3=T.concatenate([origin_input_tensor3_r,posi_emb_tensor3_r],axis=1),
                 mask_matrix = mask_matrix_r,
                 image_shape=(image_shape_r[0],image_shape_r[1],image_shape_r[2]+posi_emb_size,image_shape_r[3]),
                 filter_shape=(filter_shape[0],filter_shape[1],filter_shape[2]+posi_emb_size,filter_shape[3]), W=W_posi, b=b_posi)
        temp_conv_output_r = conv_model_r.naked_conv_out
        conv_model_weighted_l = Conv_with_Mask(rng, input_tensor3=weighted_sum_l,
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_l = conv_model_weighted_l.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_r=T.repeat(mask_matrix_r.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_r=(1.0-mask_for_conv_output_r)*(mask_for_conv_output_r-10)

        self.biased_conv_attend_out_r = T.tanh(biased_temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
        self.biased_attentive_sumpool_vec_r=T.sum(self.biased_conv_attend_out_r, axis=2)
        self.biased_attentive_meanpool_vec_r=self.biased_attentive_sumpool_vec_r/T.sum(mask_matrix_r,axis=1).dimshuffle(0,'x')
        self.conv_attend_out_r = T.tanh(temp_conv_output_r+ temp_conv_output_weighted_l+ b_posi.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
               
        masked_biased_conv_output_r=self.biased_conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_r=T.max(masked_biased_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        masked_conv_output_r=self.conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_r=T.max(masked_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size


class Conv_for_Pair_V1(object):
    """we define CNN by input tensor3 and output tensor3, like RNN, filter width must by 3,5,7..."""

    def __init__(self, rng, 
                 origin_input_tensor3, origin_input_tensor3_r, 
                 input_tensor3, input_tensor3_r, 
                 #our method
                 cm_origin_input_tensor3, cm_origin_input_tensor3_r, 
                 cm_input_tensor3, cm_input_tensor3_r,                  
                 ###########
                 mask_matrix, mask_matrix_r,
                 filter_shape, filter_shape_context, 
                 image_shape, image_shape_r, 
                 W, b, W_posi, b_posi, W_context, b_context, posi_emb_matrix, posi_emb_size, K_ratio):
        batch_size = origin_input_tensor3.shape[0]
        hidden_size = origin_input_tensor3.shape[1]

        l_len = origin_input_tensor3.shape[2]
        r_len = origin_input_tensor3_r.shape[2]
        #construct interaction matrix
        input_tensor3 = input_tensor3*mask_matrix.dimshuffle(0,'x',1)
        input_tensor3_r = input_tensor3_r*mask_matrix_r.dimshuffle(0,'x',1) #(batch, hidden, r_len)
        dot_tensor3 = T.batched_dot(input_tensor3.dimshuffle(0,2,1),input_tensor3_r) #(batch, l_len, r_len)
        # our method
        cm_input_tensor3 = cm_input_tensor3 * mask_matrix.dimshuffle(0,'x',1)
        cm_input_tensor3_r = cm_input_tensor3_r * mask_matrix_r.dimshuffle(0,'x',1) #(batch, hidden, r_len)
        cm_dot_tensor3 = T.batched_dot(cm_input_tensor3.dimshuffle(0,2,1), cm_input_tensor3_r) #(batch, l_len, r_len)
        
        new_dot_tensor3 = dot_tensor3 + 0.01 * cm_dot_tensor3
    
        '''
        try to get position shift of best match
        '''
        aligned_posi_l = T.argmax(dot_tensor3, axis=2).flatten()
        posi_emb_tensor3_l = posi_emb_matrix[aligned_posi_l].reshape((batch_size,dot_tensor3.shape[1],posi_emb_size)).dimshuffle(0,2,1) #(batch, emb_size, l_len)
        
        aligned_posi_r = T.argmax(dot_tensor3, axis=1).flatten()
        posi_emb_tensor3_r = posi_emb_matrix[aligned_posi_r].reshape((batch_size,dot_tensor3.shape[2],posi_emb_size)).dimshuffle(0,2,1) #(batch, emb_size, r_len)

        l_max_cos = (1.0 - T.min(selu(cm_dot_tensor3), axis=2))/(1.0 + 0.5 * T.max(selu(dot_tensor3), axis=2))#1.0/T.exp(T.max(T.nnet.sigmoid(dot_tensor3), axis=2)) #(batch, l_len)
        r_max_cos = (1.0 - T.min(selu(cm_dot_tensor3), axis=1))/(1.0 + 0.5 * T.max(selu(dot_tensor3), axis=1))#1.0/T.exp(T.max(T.nnet.sigmoid(dot_tensor3), axis=1)) #(batch, r_len)
        '''
        another interaction matrix
        '''

        dot_matrix_for_right = T.nnet.softmax(new_dot_tensor3.reshape((batch_size*l_len, r_len)))  #(batch*l_len, r_len)
        dot_tensor3_for_right = dot_matrix_for_right.reshape((batch_size, l_len, r_len))#(batch, l_len, r_len)
        weighted_sum_r = T.batched_dot(dot_tensor3_for_right, input_tensor3_r.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix.dimshuffle(0,'x',1) #(batch,hidden, l_len)

        dot_matrix_for_left = T.nnet.softmax(new_dot_tensor3.dimshuffle(0,2,1).reshape((batch_size*r_len, l_len))) #(batch*r_len, l_len)
        dot_tensor3_for_left = dot_matrix_for_left.reshape((batch_size, r_len, l_len))#(batch, r_len, l_len)
        
        weighted_sum_l = T.batched_dot(dot_tensor3_for_left, input_tensor3.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix_r.dimshuffle(0,'x',1) #(batch,hidden, r_len)

        #convolve left, weighted sum r
        biased_conv_model_l = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3*l_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_l = biased_conv_model_l.naked_conv_out
        self.conv_out_l = biased_conv_model_l.masked_conv_out
        self.maxpool_vec_l = biased_conv_model_l.maxpool_vec
        conv_model_l = Conv_with_Mask(rng, input_tensor3=T.concatenate([origin_input_tensor3,posi_emb_tensor3_l],axis=1),
                 mask_matrix = mask_matrix,
                 image_shape=(image_shape[0], image_shape[1], image_shape[2]+posi_emb_size, image_shape[3]),
                 filter_shape=(filter_shape[0],filter_shape[1],filter_shape[2]+posi_emb_size,filter_shape[3]), W=W_posi, b=b_posi)
        temp_conv_output_l = conv_model_l.naked_conv_out
        conv_model_weighted_r = Conv_with_Mask(rng, input_tensor3=weighted_sum_r,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_r = conv_model_weighted_r.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_l=T.repeat(mask_matrix.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_l=(1.0-mask_for_conv_output_l)*(mask_for_conv_output_l-10)

        self.biased_conv_attend_out_l = T.tanh(biased_temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        self.biased_attentive_sumpool_vec_l=T.sum(self.biased_conv_attend_out_l, axis=2)
        self.biased_attentive_meanpool_vec_l=self.biased_attentive_sumpool_vec_l/T.sum(mask_matrix,axis=1).dimshuffle(0,'x')
        masked_biased_conv_output_l=self.biased_conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_l=T.max(masked_biased_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        self.conv_attend_out_l = T.tanh(temp_conv_output_l+ temp_conv_output_weighted_r+ b_posi.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        masked_conv_output_l=self.conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_l=T.max(masked_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        #convolve right, weighted sum l
        biased_conv_model_r = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3_r*r_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_r = biased_conv_model_r.naked_conv_out
        self.conv_out_r = biased_conv_model_r.masked_conv_out
        self.maxpool_vec_r = biased_conv_model_r.maxpool_vec
        conv_model_r = Conv_with_Mask(rng, input_tensor3=T.concatenate([origin_input_tensor3_r,posi_emb_tensor3_r],axis=1),
                 mask_matrix = mask_matrix_r,
                 image_shape=(image_shape_r[0],image_shape_r[1],image_shape_r[2]+posi_emb_size,image_shape_r[3]),
                 filter_shape=(filter_shape[0],filter_shape[1],filter_shape[2]+posi_emb_size,filter_shape[3]), W=W_posi, b=b_posi)
        temp_conv_output_r = conv_model_r.naked_conv_out
        conv_model_weighted_l = Conv_with_Mask(rng, input_tensor3=weighted_sum_l,
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_l = conv_model_weighted_l.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_r=T.repeat(mask_matrix_r.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_r=(1.0-mask_for_conv_output_r)*(mask_for_conv_output_r-10)

        self.biased_conv_attend_out_r = T.tanh(biased_temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
        self.biased_attentive_sumpool_vec_r=T.sum(self.biased_conv_attend_out_r, axis=2)
        self.biased_attentive_meanpool_vec_r=self.biased_attentive_sumpool_vec_r/T.sum(mask_matrix_r,axis=1).dimshuffle(0,'x')
        self.conv_attend_out_r = T.tanh(temp_conv_output_r+ temp_conv_output_weighted_l+ b_posi.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
               
        masked_biased_conv_output_r=self.biased_conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_r=T.max(masked_biased_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        masked_conv_output_r=self.conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_r=T.max(masked_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
