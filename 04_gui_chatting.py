from tkinter import *

import time

from pickle import load

# import numpy dependencies
import numpy as np
from numpy import array
from numpy import argmax

# keras dependencies
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import pandas as pd
import win32com.client as wincl

# import nltk dependencies
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import cmudict
from nltk.translate.bleu_score import corpus_bleu

from curses.ascii import isdigit
import re
import os
import pyttsx3
import _thread as thread
import string
import random

class TalkingHead(object):
	"""docstring for TalkingHead"""
	def __init__(self):
		super(TalkingHead, self).__init__()

		self.deep_net, self.max_sentence_size, self.word_idx = load(open('model_data/deep_net_model.pkl','rb'))
		self.states_dict = {"reply" : self.reply_state_beta,"display" : self.display_state,"end" : self.end_state}
		self.knowledge_base = self.create_knowledge_base()
		self.lemmatizer = WordNetLemmatizer()
		self.books_shown = False
		self.bot_speak = self.reply_state_beta
		self.instr = 'reply'
		self.reply = "hi"
		self.next_state = 'reply'
		self.ZIRA = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0'
		self.filename = 'model_data/conversation_record.txt'
		self.file_handle = None
		# load datasets (already there, sinha's code)
		self.dataset = self.load_clean_sentences('model_data/both.pkl')
		self.dataset1= self.dataset.reshape(-1,1)
		self.all_tokenizer = self.create_tokenizer(self.dataset1[:,0])
		self.all_vocab_size = len(self.all_tokenizer.word_index) + 1
		self.all_length = self.max_length(self.dataset1[:, 0])

		# load model (already there, sinha's code)
		self.model = load_model('model_data/model_uno.h5')
		self.last_reply = ''
		self.topic_of_conv = None

		#dict and speaking defs
		# defining the text to speech engine
		self.engine = pyttsx3.init()
		self.engine.setProperty('voice',self.ZIRA)
		self.d = cmudict.dict()

		#GUI Prep
		self.window = Tk()
		self.window.winfo_toplevel().title("ZIRA")

		self.img=PhotoImage(file='f_animate/face_normal.png')
		self.label_img = Label(self.window,image=self.img, bg='white')
		self.label_img.grid(row=0,column=0,rowspan=2)
		
		self.messages = Text(self.window, width=50,wrap=WORD)
		self.messages.grid(row=0,column=1,columnspan=3,sticky=W)
		
		self.input_user = StringVar()
		self.label_field = Label(self.window, text="Enter Query",width=10)
		self.label_field.grid(row=1,column=1,columnspan=1)
		
		self.input_field = Entry(self.window, text=self.input_user, width=50)
		self.input_field.grid(row=1,column=2,columnspan=2,sticky=W)

		#self.error_button = Button(self.window,text="Expected Something Else",command=self.error_handling_function)
		#self.error_button.grid(row=2,column=2,columnspan=2,sticky=W)

		self.input_field.bind("<Return>", self.Enter_pressed)
		self.messages.bind("<Configure>", self.reset_tabstop)


	def error_handling_function(self) :
		self.new_window = Toplevel(self.window)

		self.new_input_user = StringVar()
		
		self.new_label_field = Label(self.new_window,text="Enter Correct Text",width=50)
		self.new_label_field.grid(row=0,column=0,columnspan=2,sticky=W)

		self.new_input_field = Entry(self.new_window,text=self.new_input_user,width=50)
		self.new_input_field.grid(row=1,column=0,columnspan=2,sticky=W)

		self.new_input_field.bind("<Return>",self.correct_error_function)

	def correct_error_function(self,event) :
		self.new_input_get = self.new_input_field.get()
		self.file_handle.write('Bot response Corrected ==> {}\n'.format(self.new_input_get))

		self.messages.insert(INSERT, 'Bot(Corrected)~: {}\n'.format(self.new_input_get))
		self.messages.see("end")

		self.new_window.destroy()


	def tokenize(self,text) :

		stop_words = set(stopwords.words('english'))
		raw_tokens = word_tokenize(text)
		raw_tokens = [w.lower() for w in raw_tokens]
	
		tokens = list()
	
		for tk in raw_tokens :
			tkns = tk.split('-')
			for tkn in tkns :
				tokens.append(tkn)
		
		old_punctuation = string.punctuation
		new_punctuation = old_punctuation.replace('-','')
		
		table = str.maketrans('','',new_punctuation)
		
		stripped = [w.translate(table) for w in tokens]
		
		words = [word for word in stripped if word.isalpha()]
		words = [w for w in words if not w in stop_words]
		
		return words

	def vectorize(self,tokenized_line,max_sentence_size,word_idx) :

		lq = max(0,max_sentence_size-len(tokenized_line))
		vec_line = [word_idx[w] if w in word_idx.keys() else 0 for w in tokenized_line] + lq*[0]
		vec_line = np.array(vec_line)
		return vec_line

	def create_knowledge_base(self) :
		
		books_data_base = pd.read_csv('data/booksoutnew1.csv')
		topic_set = set()
		list_of_topics = books_data_base['topic'].values
		topic_set = set(list_of_topics)
		knowledge_base = dict()
		
		for index, row in books_data_base.iterrows() :
			book_name = row['book'].strip().lower()
			book_topic = row['topic'].strip().lower()
			if book_topic in knowledge_base.keys() :
				list_of_books = set(knowledge_base[book_topic])
			else :
				list_of_books = set()
				
			list_of_books.add(book_name)
			knowledge_base[book_topic] = list(list_of_books)
			
		return knowledge_base

	def predict_text(self,text) :

		tokens = self.tokenize(text)
		vec_set = self.vectorize(tokens,self.max_sentence_size,self.word_idx)
		test_example = np.array([vec_set])
		pred = self.deep_net.predict(test_example)
		
		label = np.argmax(pred)

		return label

	def talk(self) :

		self.window.mainloop()
		
	# speak function that converts the text to speech
	def saySomething(self,txt,language):

		self.engine.say(txt)
		self.engine.runAndWait()

	def nsyl(self,word):
		return [len(list(y for y in x if isdigit(y[-1]))) for x in self.d[word.lower() if word in self.d.keys() else random.choice(list(self.d.keys()))]]
	

	# Animate function that animates the woman's face
	def animate(self,root, paragraph) :

		self.img=PhotoImage(file='f_animate/face_normal.png')
		self.label_img.configure(image=self.img)
		self.label_img.image = self.img

		#for i in range(1,5):
		#    root.update()
		root.update()

		paragraph = re.sub('[!,;?]', '.', paragraph)
		lastImage = self.img

		for sentence in paragraph.split("."):
			try:
				thread.start_new(self.saySomething,(sentence,"en",))
			except:
				pass

			workingSentence = " "
			time.sleep(0.32)
			for word in sentence.split():

				ipa = self.d[word.lower() if word in self.d.keys() else random.choice(list(self.d.keys()))]

				ipa = ipa[0]
				syl = self.nsyl(word)
				syl = syl[0]
				timePerChar = 0.20/float(len(ipa))
				for char in ipa:
					print(char, end=" ")
					if "m" in char.lower() or "b" in char.lower() or "p" in char.lower():
						self.img= PhotoImage(file='./f_animate/face_mbp.png')
					elif "th" in char.lower():
						self.img= PhotoImage(file='./f_animate/face_th.png')
					elif "ee" in char.lower():
						self.img=PhotoImage(file='./f_animate/face_ee.png')
					elif "oo" in char.lower():
						self.img=PhotoImage(file='./f_animate/face_oo.png')
					elif "l" in char.lower():
						self.img=PhotoImage(file='./f_animate/face_l.png')
					elif "f" in char.lower() or "v" in char.lower():
						self.img=PhotoImage(file='./f_animate/face_fv.png')
					elif "g" in char.lower() or "sh" in char.lower() or "ch" in char.lower():
						self.img=PhotoImage(file='./f_animate/face_gshch.png')
					elif "s" in char.lower() or "d" in char.lower() or "t" in char.lower() or "r" in char.lower() or "k" in char.lower() or "c" in char.lower():
						self.img=PhotoImage(file='./f_animate/face_sdtrck.png')
					elif "a" in char.lower():
						self.img=PhotoImage(file='./f_animate/face_a.png')
					elif "o" in char.lower():
						self.img=PhotoImage(file='./f_animate/face_o.png')
					else:
						self.img = lastImage

					self.label_img.configure(image=self.img)
					self.label_img.image = self.img
					#for i in range(1,5):
					#    root.update()
					root.update()
					self.lastImage = self.img

				self.img=PhotoImage(file='f_animate/face_normal.png')
				lastImage = self.img
				time.sleep(0.1)

			print('')
			self.img=lastImage
			self.label_img.configure(image=self.img)
			self.label_img.image = self.img
			#for i in range(1,5):
			#    root.update()

			root.update()
			time.sleep(0.4)

	# load a clean dataset
	def load_clean_sentences(self,filename):
		return load(open(filename, 'rb'))

	# fit a tokenizer
	def create_tokenizer(self,lines):
		tokenizer = Tokenizer(char_level=False)
		tokenizer.fit_on_texts(lines)
		return tokenizer

	# max sentence length
	def max_length(self,lines):
		return max(len(line.split()) for line in lines)

	# map an integer to a word
	def word_for_id(self,integer, tokenizer):
		for word, index in tokenizer.word_index.items():
			if index == integer:
				return word
		return None

	# generate target given source sequence
	def predict_sequence(self,model, tokenizer, source):
		prediction = self.model.predict(source, verbose=0)[0]
		integers = [argmax(vector) for vector in prediction]
		target = list()
		for i in integers:
			word = self.word_for_id(i, self.all_tokenizer)
			if word is None:
				break
			target.append(word)
		return ' '.join(target)

	# translate
	def translate(self,model, tokenizer, sources):
		predicted = list()
		for i, source in enumerate(sources):
			# translate encoded source text
			source = source.reshape((1, source.shape[0]))
			translation = self.predict_sequence(self.model, self.all_tokenizer, source)
			print('ANSWER: %s' % (translation))

			self.animate(self.window, translation)
			translation_1 = "~BOT : " + translation
			self.messages.insert(INSERT, '%s\n' % translation_1)
			self.messages.see("end")
			
			self.file_handle.write('{}\n'.format(translation))
			predicted.append(translation.split())
			
			return translation

	def retrieval(self,reply):

		if not self.books_shown :
			self.animate(self.window, 'You have the following options:')
			nouns_set = set()
			tokens = nltk.word_tokenize(reply)
			pos_tags = nltk.pos_tag(tokens)
			
			for word, tag in pos_tags :
				if tag == 'NN' or tag == 'NNS':
					nouns_set.add(self.lemmatizer.lemmatize(word))
			
			books_set = set()
			nouns_list = list(nouns_set)
			nouns_list.remove('book')
			print(nouns_list)
			#cnt = 1
			for key in self.knowledge_base.keys() :
				key_look = True
				for noun in nouns_list :
					if noun not in key :
						key_look = False
				if key_look == True :
					for book in self.knowledge_base[key] :
						books_set.add(">) " + book)
						#cnt = cnt + 1
			#sorted(books_set)
			
			self.book_list = list(books_set)
			self.book_list.sort()
			'''for i in range(1, len(self.book_list)):
				print(self.book_list[i])'''
			print('Length of books list is {}'.format(len(self.book_list)))
			self.book_start_index = 0
			self.book_end_index = min(self.book_start_index+10,len(self.book_list))

			books_to_show = self.book_list[self.book_start_index:self.book_end_index]
			
			bot_display_response = '~BOT : You have the following options\n \n{}\n \nwould you like to borrow any of these ?\n'.format('\n'.join(books_to_show))
			bot_write_response = 'You have the following options\n{}\n<SILENCE>\twould you like to borrow any of these ?'.format('\n'.join(books_to_show))
			bot_response = 'Would you like to borrow any of these books ?'
			self.books_shown = True

		else :

			self.book_start_index = self.book_end_index
			self.book_end_index = min(self.book_start_index + 10,len(self.book_list))

			if self.book_start_index == len(self.book_list) :

				self.books_shown = True

				bot_display_response = '~BOT : Already exhausted all books in this topic , did you find anything you like ?'
				bot_write_response = '~BOT : Already exhausted all books in this topic , did you find anything you like ?'
				bot_response = 'Already exhausted all books in this topic , did you find anything you like ?'
			else :

				books_to_show = self.book_list[self.book_start_index:self.book_end_index]
				bot_display_response = '~BOT : You have the following options\n \n{}\n \nwould you like to borrow any of these ?\n'.format('\n'.join(books_to_show))
				bot_write_response = 'You have the following options\n{}\n<SILENCE>\twould you like to borrow any of these ?'.format('\n'.join(books_to_show))
				bot_response = 'Would you like to borrow any of these books ?'
			#self.animate(self.window, bot_response)            

		#self.animate(self.window, bot_response)
		self.messages.insert(INSERT, '{}\n'.format(bot_display_response))
		self.messages.see("end")
		self.file_handle.write('{}\n'.format(bot_write_response))
		self.animate(self.window, bot_response)


	def reset_tabstop(self,event):
		event.widget.configure(tabs=(event.width-8, "right"))

	def Enter_pressed(self,event):

		self.input_get = self.input_field.get()
		self.input_get_1 = '~USER: ' + self.input_get
		self.messages.insert(INSERT, '%s\n' % self.input_get_1)
		self.messages.see("end")


		if not self.file_handle :
			if not os.path.exists(self.filename) :
				self.file_handle = open(self.filename,'w')
			else :
				self.file_handle = open(self.filename,'a')

		self.file_handle.write('{}\t'.format(self.input_get))
		self.reply = self.predict_utterance(self.input_get)

		self.input_user.set('')
		return "break"

	def predict_utterance(self,query) :

		print('Query is : {}'.format(query))
		if(self.next_state!='reply'):

			self.instr = self.find_instr(query)
			print('\nInstruction found is :{}'.format(self.instr))

		self.next_state, bot_response = self.bot_speak(self.instr,query)

		print('BOT RESP#\t',bot_response)
		self.bot_speak = self.states_dict[self.next_state]
		self.last_reply = bot_response

		print('\nnext_state: '+self.next_state)

		return bot_response

	def reset_global_variables(self) :

		self.topic_of_conv = None
		self.books_shown = False

		if not self.file_handle :
			if not os.path.exists(self.filename) :
				self.file_handle = open(self.filename,'w')
			else :
				self.file_handle = open(self.filename,'a')

	def reply_state_beta(self,instr,query) :

		print('************ reply beta **************')

		self.reset_global_variables()

		if query == 'bye' or query=='sure' or query=='alright' or query=='waiting' or query== '<SILENCE>' or query=='silence':
			'''if self.topic_of_conv :
				query = self.topic_of_conv
			else :
				self.topic_of_conv = query'''
			next_state, bot_response = self.display_state('display',query)   
			#print('BOT ANSWER: ' + bot_response) 
		else :
			query = query.strip().split('\n')
			X = self.all_tokenizer.texts_to_sequences(query)
			X = pad_sequences(X, maxlen=self.all_length, padding='post')
			
			# find reply and print it out
			bot_response = self.translate(self.model, self.all_tokenizer, X)
			next_state = "reply"
		return next_state, bot_response

	def display_state(self,instr,query) :
		print('************ display **************')
		query = query.strip().split('\n')
		
		if instr == "end" :
			bot_response = "thanks for using the bot"
			bot_display_response = bot_response
			next_state = "end"
			self.animate(self.window, bot_response)
			self.messages.insert(INSERT, '{}\n'.format(bot_display_response))
			self.file_handle.write('{}\n'.format(bot_display_response))

			# give spacing between the dialogs
			self.file_handle.write('\n')
			self.file_handle.close()
			self.file_handle = None
			self.messages.see("end")

		elif instr == "reply" :
			X = self.all_tokenizer.texts_to_sequences(query)
			X = pad_sequences(X, maxlen=self.all_length, padding='post')
			self.topic_of_conv = None

			# find reply and print it out
			bot_response = self.translate(self.model, self.all_tokenizer, X)
			next_state = "reply"
		else :
			#print('toc: '+ self.topic_of_conv)
			if self.topic_of_conv :
				query = self.topic_of_conv
			else :
				self.topic_of_conv = self.last_reply
			bot_response = self.retrieval(self.topic_of_conv)
			next_state = "display"
			#bot_response = ''
		return next_state, bot_response

	def end_state(self,instr,query) :
		print('************ end **************')
		query = query.strip().split('\n')
		self.messages.delete('1.0',END)
		self.reset_global_variables()
		if instr == "reply" :
			X = self.all_tokenizer.texts_to_sequences(query)
			X = pad_sequences(X, maxlen=self.all_length, padding='post')
			# find reply and print it out
			bot_response = self.translate(self.model, self.all_tokenizer, X)
			bot_display_response = bot_response
			next_state = "reply"
		else :
			bot_response = "conversation is over!!"
			bot_display_response = bot_response
			next_state = "reply"
			self.animate(self.window, bot_response)
			self.messages.insert(INSERT, '{}\n'.format(bot_display_response))
			self.messages.see("end")
		return next_state, bot_response


	# find the instruction to move the FSM given the input query
	def find_instr(self,query) :

		self.instr_states_dict = {1:'end',2:'display',3:'reply'}
		label_predicted = self.predict_text(query)
		print('label predicted is {}'.format(label_predicted))
		instr = self.instr_states_dict[label_predicted]
		return instr




if __name__ == '__main__' :
	#some variables
	t_head = TalkingHead()
	t_head.talk()
