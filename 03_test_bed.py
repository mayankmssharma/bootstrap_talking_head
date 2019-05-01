from tkinter import *
import time
from pickle import load
from numpy import array, argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
import nltk
from nltk.corpus import cmudict
from curses.ascii import isdigit
import re
import pyttsx3
import _thread as thread
from autocorrect import spell

def spell_check(inp):
    s_1 = ""
    for word in inp.split(" "):
        s_1+=str(spell(word))
        s_1+=str(" ")
    return s_1[:len(s_1)-1]

def saySomething(txt,language):
    engine.say(txt)
    engine.runAndWait()

def nsyl(word):
    return [len(list(y for y in x if isdigit(y[-1]))) for x in d[word.lower()]]
    
def animate(root, paragraph) :
    img=PhotoImage(file='f_animate/face_normal.png')
    label_img.configure(image=img)
    label_img.image = img
    #for i in range(1,5):
    root.update()
    #paragraph =  '''Hello there! So you want me to be in the Student Library ?'''
    paragraph = re.sub('[!,;?]', '.', paragraph)
    lastImage = img

    for sentence in paragraph.split("."):
        try:
            #print('here 1')
            thread.start_new(saySomething,(sentence,"en",))
        except:
            pass
        workingSentence = " "
        time.sleep(0.32)
        for word in sentence.split():
            #print(word)
            ipa = d[word.lower()]
            ipa = ipa[0]
            syl = nsyl(word)
            syl = syl[0]
            timePerChar = 0.20/float(len(ipa))
            for char in ipa:
                print(char, end=" ")
                if "m" in char.lower() or "b" in char.lower() or "p" in char.lower():
                    img= PhotoImage(file='./f_animate/face_mbp.png')
                elif "th" in char.lower():
                    img= PhotoImage(file='./f_animate/face_th.png')
                elif "ee" in char.lower():
                    img=PhotoImage(file='./f_animate/face_ee.png')
                elif "oo" in char.lower():
                    img=PhotoImage(file='./f_animate/face_oo.png')
                elif "l" in char.lower():
                    img=PhotoImage(file='./f_animate/face_l.png')
                elif "f" in char.lower() or "v" in char.lower():
                    img=PhotoImage(file='./f_animate/face_fv.png')
                elif "g" in char.lower() or "sh" in char.lower() or "ch" in char.lower():
                    img=PhotoImage(file='./f_animate/face_gshch.png')
                elif "s" in char.lower() or "d" in char.lower() or "t" in char.lower() or "r" in char.lower() or "k" in char.lower() or "c" in char.lower():
                    img=PhotoImage(file='./f_animate/face_sdtrck.png')
                elif "a" in char.lower():
                    img=PhotoImage(file='./f_animate/face_a.png')
                elif "o" in char.lower():
                    img=PhotoImage(file='./f_animate/face_o.png')
                else:
                    img = lastImage

                label_img.configure(image=img)
                label_img.image = img
                #for i in range(1,5):
                root.update()
                lastImage = img

            img=PhotoImage(file='f_animate/face_normal.png')
            lastImage = img
            time.sleep(0.1)

        print('')
        img=lastImage
        label_img.configure(image=img)
        label_img.image = img
        #for i in range(1,5):
        root.update()
        time.sleep(0.4)

# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(lines)
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

# translate
def translate(model, tokenizer, sources):
    predicted = list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, all_tokenizer, source)
        print('ANSWER: %s' % (translation))
        animate(window, translation)
        translation_1 = "~BOT : " + translation
        messages.insert(INSERT, '%s\n' % translation_1)
        messages.see("end")
        predicted.append(translation.split())
        return translation

def retrieval(reply):
    #print("Retrieval function called: " + reply)
    data=pd.read_csv('data/booksoutnew1.csv')
    arrtopic=data['topic']
    arrbook=data['book']
    #arrtopic=data['topic']
    for i in range(len(arrtopic)):
        arrtopic[i]=arrtopic[i].lower()
    arrbook=data['book']
    replsplit=reply.split()
    
    topics=["networks","software engineering","theory of computation","information systems","information security","computer architecture","signal processing","logic in computer science","database systems","machine learning","artificial intelligence","algorithm design"]
    for topicvals in topics:
        topic=topicvals.split()
        #print("topic 0 is", topic[0])
    #topic=arrtopic[0].split();
        if topic[0].lower() in replsplit:
            #print('found')
            #print('ANSWER: The following options are available:\n\n')
            messages.insert(INSERT, '%s\n' % '~BOT : The following options are available:\n')
            #print(len(arrtopic))
            cnt = 0
            for i in range(len(arrtopic)):
                if(cnt==5):
                    break
                if (arrtopic[i].split())[0]==topic[0]:
                    cnt = cnt + 1
                    temp = str(cnt) + ") " + arrbook[i]
                    messages.insert(INSERT, '%s\n' % temp)
                    messages.see("end")
                    #print("       ", arrbook[i])
    #print("ANSWER: Would you like to borrow any of these books?")
    animate(window, 'Would you like to borrow any of these books')
    messages.insert(INSERT, '%s\n' % '\n~BOT : Would you like to borrow any of these books?')
    messages.see("end")
    '''while(True):
        #q=(input(str("YOU: ")))
        if( q!=None and ((q.split())[0]=='yes' or (q.split())[0]=='sure' or (q.split())[0]=='alright') ):
            print("ANSWER: The book has been issued. Thanks for using")
            break
        else:
            print("ANSWER: Query completed. Thanks for using")
            break'''


def reset_tabstop(event):
    event.widget.configure(tabs=(event.width-8, "right"))

def initiate_reply(q):
    print(q)
    if q == 'bye' or q=='sure' or q=='alright' or q=='waiting' or q== '<SILENCE>' or q=='silence':
        #print("Last verse was:", reply)
        retrieval(reply)
        return
    elif( q!=None and ((q.split())[0]=='yes') ):
        animate(window, 'Query completed Thanks for using')
        messages.insert(INSERT, '%s\n' % '~BOT : Query completed. Thanks for using.')
        messages.insert(INSERT, '%s\n' % '-------------------------------------------\n')
        messages.see("end")
        return
    q = q.strip().split('\n')

    #we tokenize
    X = all_tokenizer.texts_to_sequences(q)
    X = pad_sequences(X, maxlen=all_length, padding='post')
        
    # find reply and print it out
    return translate(model, all_tokenizer, X)

def Enter_pressed(event):
    global reply
    input_get = input_field.get()
    #input_get = spell_check(input_get)
    input_get_1 = '~USER: ' + input_get
    messages.insert(INSERT, '%s\n' % input_get_1)
    messages.see("end")

    reply = initiate_reply(input_get)
    input_user.set('')
    return "break"


if __name__ == '__main__' :
    #some variables
    reply = "hi"
    ZIRA = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0'
    # load datasets
    dataset = load_clean_sentences('model_data/both.pkl')
    dataset1=dataset.reshape(-1,1)

    # prepare tokenizer
    all_tokenizer = create_tokenizer(dataset1[:,0])
    all_vocab_size = len(all_tokenizer.word_index) + 1
    all_length = max_length(dataset1[:, 0])
    model = load_model('model_data/model_uno.h5')

    #dict and speaking defs
    engine = pyttsx3.init()
    engine.setProperty('voice',ZIRA)
    d = cmudict.dict()

    #GUI Prep
    window = Tk()
    window.winfo_toplevel().title("Talking Head")

    img=PhotoImage(file='f_animate/face_normal.png')
    label_img = Label(window,image=img, bg='white')
    label_img.grid(row=0,column=0,rowspan=2)
    #window.update()
    #label_img.pack(side=LEFT)
    
    messages = Text(window, width=50,wrap=WORD)
    messages.grid(row=0,column=1,columnspan=3,sticky=W)
    #messages.pack()
    
    input_user = StringVar()
    label_field = Label(window, text="Enter Query",width=10)
    label_field.grid(row=1,column=1,columnspan=1)
    #label_field.pack(side=LEFT)
    
    input_field = Entry(window, text=input_user, width=50)
    input_field.grid(row=1,column=2,columnspan=2,sticky=W)
    #input_field.pack(side=BOTTOM, fill=X)
    
    input_field.bind("<Return>", Enter_pressed)
    messages.bind("<Configure>", reset_tabstop)
    window.mainloop()
