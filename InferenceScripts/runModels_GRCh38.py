import sys, getopt, os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import official.nlp
import official.nlp.keras_nlp.layers
from transformers import TFT5EncoderModel, T5Tokenizer,T5Config

def MaverickArchitecture1(input_shape=201,classes=3,classifier_activation='softmax',**kwargs):
    input0 = tf.keras.layers.Input(shape=(input_shape,51),name='mm_orig_seq')
    input1 = tf.keras.layers.Input(shape=(input_shape,51),name='mm_alt_seq')
    input2 = tf.keras.layers.Input(shape=12,name='non_seq_info')

    # project input to an embedding size that is easier to work with
    x_orig = tf.keras.layers.experimental.EinsumDense('...x,xy->...y',output_shape=64,bias_axes='y')(input0)
    x_alt = tf.keras.layers.experimental.EinsumDense('...x,xy->...y',output_shape=64,bias_axes='y')(input1)

    posEnc_wt = official.nlp.keras_nlp.layers.PositionEmbedding(max_length=input_shape)(x_orig)
    x_orig = tf.keras.layers.Masking()(x_orig)
    x_orig = tf.keras.layers.Add()([x_orig,posEnc_wt])
    x_orig = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,dtype=tf.float32)(x_orig)
    x_orig = tf.keras.layers.Dropout(0.05)(x_orig)

    posEnc_alt = official.nlp.keras_nlp.layers.PositionEmbedding(max_length=input_shape)(x_alt)
    x_alt = tf.keras.layers.Masking()(x_alt)
    x_alt = tf.keras.layers.Add()([x_alt,posEnc_alt])
    x_alt = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,dtype=tf.float32)(x_alt)
    x_alt = tf.keras.layers.Dropout(0.05)(x_alt)

    transformer1 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer2 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer3 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer4 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer5 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer6 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    
    x_orig = transformer1(x_orig)
    x_orig = transformer2(x_orig)
    x_orig = transformer3(x_orig)
    x_orig = transformer4(x_orig)
    x_orig = transformer5(x_orig)
    x_orig = transformer6(x_orig)
    
    x_alt = transformer1(x_alt)
    x_alt = transformer2(x_alt)
    x_alt = transformer3(x_alt)
    x_alt = transformer4(x_alt)
    x_alt = transformer5(x_alt)
    x_alt = transformer6(x_alt)

    first_token_tensor_orig = (tf.keras.layers.Lambda(lambda a: tf.squeeze(a[:, 100:101, :], axis=1))(x_orig))
    x_orig = tf.keras.layers.Dense(units=64,activation='tanh')(first_token_tensor_orig)
    x_orig = tf.keras.layers.Dropout(0.05)(x_orig)

    first_token_tensor_alt = (tf.keras.layers.Lambda(lambda a: tf.squeeze(a[:, 100:101, :], axis=1))(x_alt))
    x_alt = tf.keras.layers.Dense(units=64,activation='tanh')(first_token_tensor_alt)
    x_alt = tf.keras.layers.Dropout(0.05)(x_alt)

    diff = tf.keras.layers.Subtract()([x_alt,x_orig])
    combined = tf.keras.layers.concatenate([x_alt,diff])

    input2Dense1 = tf.keras.layers.Dense(64,activation='relu')(input2)
    input2Dense1 = tf.keras.layers.Dropout(0.05)(input2Dense1)
    x = tf.keras.layers.concatenate([combined,input2Dense1])
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(512,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(64,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(classes, activation=classifier_activation,name='output')(x)
    model = tf.keras.Model(inputs=[input0,input1,input2],outputs=x)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.85)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def MaverickArchitecture2(input_shape=201,embeddingSize=1024,mmSize=51,classes=3,classifier_activation='softmax',**kwargs):
    input0 = tf.keras.layers.Input(shape=(input_shape,mmSize),name='alt_cons')
    input1 = tf.keras.layers.Input(shape=(input_shape,embeddingSize),name='alt_emb')
    input2 = tf.keras.layers.Input(shape=12,name='non_seq_info')

    # project input to an embedding size that is easier to work with
    alt_cons = tf.keras.layers.experimental.EinsumDense('...x,xy->...y',output_shape=64,bias_axes='y')(input0)

    posEnc_alt = official.nlp.keras_nlp.layers.PositionEmbedding(max_length=input_shape)(alt_cons)
    alt_cons = tf.keras.layers.Masking()(alt_cons)
    alt_cons = tf.keras.layers.Add()([alt_cons,posEnc_alt])
    alt_cons = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,dtype=tf.float32)(alt_cons)
    alt_cons = tf.keras.layers.Dropout(0.05)(alt_cons)

    transformer1 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer2 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer3 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer4 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer5 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    transformer6 = official.nlp.keras_nlp.layers.TransformerEncoderBlock(16,256,tf.keras.activations.relu,output_dropout=0.1,attention_dropout=0.1)
    
    alt_cons = transformer1(alt_cons)
    alt_cons = transformer2(alt_cons)
    alt_cons = transformer3(alt_cons)
    alt_cons = transformer4(alt_cons)
    alt_cons = transformer5(alt_cons)
    alt_cons = transformer6(alt_cons)

    first_token_tensor_alt = (tf.keras.layers.Lambda(lambda a: tf.squeeze(a[:, 100:101, :], axis=1))(alt_cons))
    alt_cons = tf.keras.layers.Dense(units=64,activation='tanh')(first_token_tensor_alt)
    alt_cons = tf.keras.layers.Dropout(0.05)(alt_cons)

    sharedLSTM1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.5))

    alt_emb=sharedLSTM1(input1)
    alt_emb=tf.keras.layers.Dropout(0.2)(alt_emb)

    structured = tf.keras.layers.Dense(64,activation='relu')(input2)
    structured = tf.keras.layers.Dropout(0.05)(structured)
    x = tf.keras.layers.concatenate([alt_cons,alt_emb,structured])
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(512,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(64,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(classes, activation=classifier_activation,name='output')(x)
    model = tf.keras.Model(inputs=[input0,input1,input2],outputs=x)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.85)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, dataFrameIn, tokenizer, T5Model, batch_size=32, padding=100, n_channels_emb=1024, n_channels_mm=51, n_classes=3, shuffle=True):
        self.padding = padding
        self.dim = self.padding + self.padding + 1
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels_emb = n_channels_emb
        self.n_channels_mm = n_channels_mm
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dataFrameIn=dataFrameIn
        self.tokenizer = tokenizer
        self.T5Model = T5Model
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if (len(self.list_IDs) % self.batch_size) == 0:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (((len(self.list_IDs) % self.batch_size) != 0) & (((index+1)*self.batch_size)>len(self.list_IDs))):
            indexes = self.indexes[index*self.batch_size:]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        thisBatchSize=len(list_IDs_temp)
        altEmbeddings=np.zeros((thisBatchSize, self.dim, self.n_channels_emb))
        mm_alt=np.zeros((thisBatchSize, self.dim, self.n_channels_mm))
        mm_orig=np.zeros((thisBatchSize, self.dim, self.n_channels_mm))
        nonSeq=np.zeros((thisBatchSize, 12))
        y = np.empty((thisBatchSize), dtype=int)
        AMINO_ACIDS = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19} 
        T5AltSeqTokens=[]

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # process Alt seq with T5 model to create embeddings
            transcriptID=self.dataFrameIn.loc[ID,'TranscriptID']
            changePos=self.dataFrameIn.loc[ID,'ChangePos']-1
            if changePos<0:
                changePos=0
            AltSeq=self.dataFrameIn.loc[ID,'AltSeq']
            if AltSeq[-1]!="*":
                AltSeq=AltSeq + "*"
            seqLenAlt=len(AltSeq)-1
            startPos=0
            if changePos>self.padding:
                if (changePos+self.padding)<seqLenAlt:
                    startPos=changePos-self.padding
                elif seqLenAlt>=self.dim:
                    startPos=seqLenAlt-self.dim
            endPos=changePos+self.padding
            if changePos<self.padding:
                if self.dim<seqLenAlt:
                    endPos=self.dim
                else:
                    endPos=seqLenAlt
            elif (changePos+self.padding)>=seqLenAlt:
                endPos=seqLenAlt
            T5AltSeqTokens.append(" ".join(AltSeq[startPos:endPos]))
            # prep the WT seq too
            WTSeq=self.dataFrameIn.loc[ID,'WildtypeSeq']
            if WTSeq[-1]!="*":
                WTSeq=WTSeq + "*"
            seqLen=len(WTSeq)-1
            startPos=0
            if changePos>self.padding:
                if (changePos+self.padding)<seqLen:
                    startPos=int(changePos-self.padding)
                elif seqLen>=self.dim:
                    startPos=int(seqLen-self.dim)
            endPos=int(changePos+self.padding)
            if changePos<self.padding:
                if self.dim<seqLen:
                    endPos=int(self.dim)
                else:
                    endPos=int(seqLen)
            elif (changePos+self.padding)>=seqLen:
                endPos=int(seqLen)
            T5AltSeqTokens.append(" ".join(WTSeq[startPos:endPos]))


            # collect MMSeqs WT info
            tmp=np.load("HHMFiles/" + transcriptID + "_MMSeqsProfile.npz",allow_pickle=True)
            tmp=tmp['arr_0']
            seqLen=tmp.shape[0]
            startPos=changePos-self.padding
            endPos=changePos+self.padding + 1
            startOffset=0
            endOffset=self.dim
            if changePos<self.padding:
                startPos=0
                startOffset=self.padding-changePos
            if (changePos + self.padding) >= seqLen:
                endPos=seqLen
                endOffset=self.padding + seqLen - changePos
            mm_orig[i,startOffset:endOffset,:] = tmp[startPos:endPos,:]

            # collect MMSeqs Alt info
            # change the amino acid at 'ChangePos' and any after that if needed
            varType=self.dataFrameIn.loc[ID,'varType']
            WTSeq=self.dataFrameIn.loc[ID,'WildtypeSeq']
            if varType=='nonsynonymous SNV':
                if changePos==0:
                    # then this transcript is ablated
                    altEncoded=np.zeros((seqLen,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    altEncoded[:,0:20]=0
                    altEncoded[:,50]=0
                else:
                    # change the single amino acid
                    altEncoded=np.zeros((seqLen,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    altEncoded[changePos,AMINO_ACIDS[WTSeq[changePos]]]=0
                    altEncoded[changePos,AMINO_ACIDS[AltSeq[changePos]]]=1
            elif varType=='stopgain':
                if changePos==0:
                    # then this transcript is ablated
                    altEncoded=np.zeros((seqLen,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    altEncoded[:,0:20]=0
                    altEncoded[:,50]=0
                elif seqLenAlt>seqLen:
                    altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    for j in range(changePos,seqLen):
                        altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                        altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                    for j in range(seqLen,seqLenAlt):
                        altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                    altEncoded[seqLen:,50]=1
                else:
                    altEncoded=np.zeros((seqLen,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    altEncoded[changePos:,0:20]=0
                    altEncoded[changePos:,50]=0
            elif varType=='stoploss':
                altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                altEncoded[:seqLen,:]=tmp
                for j in range(seqLen,seqLenAlt):
                    altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                altEncoded[seqLen:,50]=1
            elif varType=='synonymous SNV':
                # no change
                altEncoded=tmp
            elif ((varType=='frameshift deletion') | (varType=='frameshift insertion') | (varType=='frameshift substitution')):
                if seqLen<seqLenAlt:
                    altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    for j in range(changePos,seqLen):
                        altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                        altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                    for j in range(seqLen,seqLenAlt):
                        altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                    altEncoded[seqLen:,50]=1
                elif seqLen>seqLenAlt:
                    for j in range(changePos,seqLenAlt):
                        tmp[j,AMINO_ACIDS[WTSeq[j]]]=0
                        tmp[j,AMINO_ACIDS[AltSeq[j]]]=1
                    for j in range(seqLenAlt,seqLen):
                        tmp[j,AMINO_ACIDS[WTSeq[j]]]=0
                    altEncoded=tmp
                elif seqLen==seqLenAlt:
                    for j in range(changePos,seqLen):
                        tmp[j,AMINO_ACIDS[WTSeq[j]]]=0
                        tmp[j,AMINO_ACIDS[AltSeq[j]]]=1
                    altEncoded=tmp
                else:
                    print('Error: seqLen comparisons did not work')
                    exit()
            elif varType=='nonframeshift deletion':
                # how many amino acids deleted?
                altNucLen=0
                if self.dataFrameIn.loc[ID,'alt']!='-':
                    altNucLen=len(self.dataFrameIn.loc[ID,'alt'])
                refNucLen=len(self.dataFrameIn.loc[ID,'ref'])
                numAADel=int((refNucLen-altNucLen)/3)
                if (seqLen-numAADel)==seqLenAlt:
                    # non-frameshift deletion
                    for j in range(changePos,(changePos+numAADel)):
                        tmp[j,:20]=0
                    altEncoded=tmp
                elif seqLen>=seqLenAlt:
                    # early truncation
                    altEncoded=np.zeros((seqLen,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    for j in range(changePos,seqLenAlt):
                        altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                        altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                    altEncoded[seqLenAlt:,0:20]=0
                    altEncoded[seqLenAlt:,50]=0
                elif seqLen<seqLenAlt:
                    # deletion causes stop-loss
                    altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    for j in range(changePos,seqLen):
                        altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                        altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                    altEncoded[seqLen:,0:20]=0
                    altEncoded[seqLen:,50]=0
                else:
                    print('Error: seqLen comparisons did not work for nonframeshift deletion')
                    exit()
            elif varType=='nonframeshift insertion':
                # how many amino acids inserted?
                refNucLen=0
                if self.dataFrameIn.loc[ID,'ref']!='-':
                    altNucLen=len(self.dataFrameIn.loc[ID,'ref'])
                altNucLen=len(self.dataFrameIn.loc[ID,'alt'])
                numAAIns=int((altNucLen-refNucLen)/3)
                if (seqLen+numAAIns)==seqLenAlt:
                    # non-frameshift insertion
                    altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                    altEncoded[:changePos,:]=tmp[:changePos,:]
                    altEncoded[(changePos+numAAIns):,:]=tmp[changePos:,:]
                    for j in range(numAAIns):
                        altEncoded[(changePos+j),AMINO_ACIDS[AltSeq[(changePos+j)]]]=1
                    altEncoded[:,50]=1
                elif seqLen<seqLenAlt:
                    # stop loss
                    altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    for j in range(changePos,seqLen):
                        altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                        altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                    for j in range(seqLen,seqLenAlt):
                        altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                    altEncoded[seqLen:,50]=1
                elif seqLen>=seqLenAlt:
                    # stop gain
                    altEncoded=np.zeros((seqLen,self.n_channels_mm))
                    altEncoded[:seqLen,:]=tmp
                    for j in range(changePos,seqLenAlt):
                        altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                        altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                    altEncoded[seqLenAlt:,0:20]=0
                    altEncoded[seqLenAlt:,50]=0
                else:
                    print('Error: seqLen comparisons did not work for nonframeshift insertion')
                    exit()
            elif varType=='nonframeshift substitution':
                # is this an insertion or a deletion?
                # note that there will not be any '-' symbols in these ref or alt fields because it is a substitution
                refNucLen=len(self.dataFrameIn.loc[ID,'ref'])
                altNucLen=len(self.dataFrameIn.loc[ID,'alt'])
                if refNucLen>altNucLen:
                    # deletion
                    # does this cause an early truncation or non-frameshift deletion?
                    if seqLen>seqLenAlt: 
                        numAADel=int((refNucLen-altNucLen)/3)
                        if (seqLen-numAADel)==seqLenAlt:
                            # non-frameshift deletion
                            for j in range(changePos,(changePos+numAADel)):
                                tmp[j,:20]=0
                            altEncoded=tmp
                        else:
                            # early truncation
                            altEncoded=np.zeros((seqLen,self.n_channels_mm))
                            altEncoded[:seqLen,:]=tmp
                            for j in range(changePos,seqLenAlt):
                                altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                                altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                            altEncoded[seqLenAlt:,0:20]=0
                            altEncoded[seqLenAlt:,50]=0
                    # does this cause a stop loss?
                    elif seqLen<seqLenAlt:
                        altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                        altEncoded[:seqLen,:]=tmp
                        for j in range(changePos,seqLen):
                            altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                            altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                        for j in range(seqLen,seqLenAlt):
                            altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                        altEncoded[seqLen:,50]=1
                    else: # not sure how this would happen
                        altEncoded=np.zeros((seqLen,self.n_channels_mm))
                        altEncoded[:seqLen,:]=tmp
                        for j in range(changePos,seqLen):
                            altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                            altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                elif refNucLen<altNucLen:
                    # insertion
                    # does this cause a stop loss or non-frameshift insertion?
                    if seqLen<seqLenAlt: 
                        numAAIns=int((altNucLen-refNucLen)/3)
                        if (seqLen+numAAIns)==seqLenAlt:
                            # non-frameshift insertion
                            altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                            altEncoded[:changePos,:]=tmp[:changePos,:]
                            altEncoded[(changePos+numAAIns):,:]=tmp[changePos:,:]
                            for j in range(numAAIns):
                                altEncoded[(changePos+j),AMINO_ACIDS[AltSeq[(changePos+j)]]]=1
                            altEncoded[:,50]=1
                        else:
                            # stop loss
                            altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                            altEncoded[:seqLen,:]=tmp
                            for j in range(changePos,seqLen):
                                altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                                altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                            for j in range(seqLen,seqLenAlt):
                                altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                            altEncoded[:,50]=1
                    # does this cause an early truncation?
                    elif seqLen>seqLenAlt: 
                        altEncoded=np.zeros((seqLen,self.n_channels_mm))
                        altEncoded[:seqLen,:]=tmp
                        for j in range(changePos,seqLenAlt):
                            altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                            altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                        altEncoded[seqLenAlt:,0:20]=0
                        altEncoded[seqLenAlt:,50]=0
                    else: # not sure how this would happen, but just in case
                        altEncoded=np.zeros((seqLen,self.n_channels_mm))
                        altEncoded[:seqLen,:]=tmp
                        for j in range(changePos,seqLen):
                            altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                            altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                elif refNucLen==altNucLen:
                    if seqLen==seqLenAlt:
                        # synonymous or nonsynonymous change
                        altEncoded=np.zeros((seqLen,self.n_channels_mm))
                        altEncoded[:seqLen,:]=tmp
                        altEncoded[changePos,AMINO_ACIDS[WTSeq[changePos]]]=0
                        altEncoded[changePos,AMINO_ACIDS[AltSeq[changePos]]]=1
                    elif seqLen>seqLenAlt:
                        # early truncation
                        altEncoded=np.zeros((seqLen,self.n_channels_mm))
                        altEncoded[:seqLen,:]=tmp
                        for j in range(changePos,seqLenAlt):
                            altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                            altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                        altEncoded[seqLenAlt:,0:20]=0
                        altEncoded[seqLenAlt:,50]=0
                    elif seqLen<seqLenAlt:
                        # stop loss
                        altEncoded=np.zeros((seqLenAlt,self.n_channels_mm))
                        altEncoded[:seqLen,:]=tmp
                        for j in range(changePos,seqLen):
                            altEncoded[j,AMINO_ACIDS[WTSeq[j]]]=0
                            altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                        for j in range(seqLen,seqLenAlt):
                            altEncoded[j,AMINO_ACIDS[AltSeq[j]]]=1
                        altEncoded[seqLen:,50]=1
                    else:
                        print('non-frameshift substitution comparisons failed')
                        exit()
                else:
                    print('Error: nonframeshift substitution nucleotide length comparison did not work')
                    exit()
            startPos=changePos-self.padding
            endPos=changePos+self.padding+1
            startOffset=0
            endOffset=self.dim
            if changePos<self.padding:
                startPos=0
                startOffset=self.padding-changePos
            if (changePos + self.padding) >= seqLenAlt:
                endPos=seqLenAlt
                endOffset=self.padding + seqLenAlt - changePos
            # exception to deal with start loss SNVs that create new frameshifted products longer than the original protein (when original was shorter than padding length)
            if ((changePos==0) & (self.padding>=seqLen) & (seqLen<seqLenAlt) & (varType=='nonsynonymous SNV')):
                endPos=seqLen
                endOffset=self.padding + seqLen - changePos
            elif ((changePos==0) & (varType=='stopgain')): # related exception for stopgains at position 0
                if (seqLen+self.padding)<=self.dim:
                    endPos=seqLen
                    endOffset=self.padding + seqLen - changePos
                else:
                    endPos=self.padding+1
                    endOffset=self.dim
            mm_alt[i,startOffset:endOffset,:] = altEncoded[startPos:endPos,:]


            # non-seq info
            nonSeq[i] = self.dataFrameIn.loc[ID,['controls_AF','controls_nhomalt','pLI','pNull','pRec','mis_z','lof_z','CCR','GDI','pext','RVIS_ExAC_0.05','gerp']]
            
            # Store class
            y[i] = self.labels[ID]

        # process the altSeq and wtSeq through the T5 tokenizer (for consistency with pre-computed data used for training)
        allTokens=self.tokenizer.batch_encode_plus(T5AltSeqTokens,add_special_tokens=True, padding=True, return_tensors="tf")
        input_ids=allTokens['input_ids'][::2]
        attnMask=allTokens['attention_mask'][::2]
        # but only process the altSeq through the T5 model
        embeddings=self.T5Model(input_ids,attention_mask=attnMask)
        allEmbeddings=np.asarray(embeddings.last_hidden_state)
        for i in range(thisBatchSize):
            seq_len = (np.asarray(attnMask)[i] == 1).sum()
            seq_emb = allEmbeddings[i][1:seq_len-1]
            altEmbeddings[i,:seq_emb.shape[0],:]=seq_emb


        X={'alt_cons':mm_alt,'alt_emb':altEmbeddings,'non_seq_info':nonSeq,'mm_orig_seq':mm_orig}

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def main ( argv ):
    inBase = ''
    try:
        opts, args = getopt.getopt(argv,"h",["inputBase=","help"])
    except getopt.GetoptError:
        print('runModels.py --inputBase <baseName>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('--inputBase'):
            inBase=arg
        elif opt in ('-h','--help'):
            print('runModels.py --inputBase <baseName>')
            sys.exit()
        else:
            print('runModels.py --inputBase <baseName>')
            sys.exit()

    import pandas
    pandas.options.mode.chained_assignment = None
    from sklearn.preprocessing import QuantileTransformer
    import numpy as np
    import scipy
    from scipy.stats import rankdata
    from sklearn.metrics import classification_report
    import tensorflow as tf
    import os
    from datetime import datetime
    from sklearn.utils import resample

    batchSize=128
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_bfd", do_lower_case=False,local_files_only=True)
    T5Model = TFT5EncoderModel.from_pretrained("prot_t5_xl_bfd",local_files_only=True)


    # calculate medians and quantiles from training data
    trainingData=pandas.read_csv('trainingSet_v4.groomed_withExtraInfo2_corrected.txt',sep='\t',low_memory=False)
    trainingData.loc[trainingData['GDI']>2000,'GDI']=2000
    trainingDataNonSeqInfo=trainingData[['controls_AF','controls_nhomalt','pLI','pNull','pRec','mis_z','lof_z','CCR','GDI','pext','RVIS_ExAC_0.05','gerp']].copy(deep=True)
    trainingDataNonSeqInfo.loc[trainingDataNonSeqInfo['controls_AF'].isna(),'controls_AF']=0
    trainingDataNonSeqInfo.loc[trainingDataNonSeqInfo['controls_nhomalt'].isna(),'controls_nhomalt']=0
    trainingDataNonSeqInfo.loc[trainingDataNonSeqInfo['controls_nhomalt']>10,'controls_nhomalt']=10
    trainingDataNonSeqMedians=trainingDataNonSeqInfo.median()
    trainingDataNonSeqInfo=trainingDataNonSeqInfo.fillna(trainingDataNonSeqMedians)
    trainingDataNonSeqInfo=np.asarray(trainingDataNonSeqInfo.to_numpy()).astype(np.float32)

    # scale columns by QT
    qt = QuantileTransformer(subsample=1e6, random_state=0, output_distribution='uniform')
    qt=qt.fit(trainingDataNonSeqInfo)
    trainingDataNonSeqInfo=qt.transform(trainingDataNonSeqInfo)

    # load the models
    Arc1Model1 = MaverickArchitecture1()
    Arc1Model1.load_weights('weights_TransformerNetDiff_model_1')
    Arc1Model2 = MaverickArchitecture1()
    Arc1Model2.load_weights('weights_TransformerNetDiff_classWeights_1_2_7_model_1')
    Arc1Model3 = MaverickArchitecture1()
    Arc1Model3.load_weights('weights_TransformerNetDiff_classWeights_1_2_7_model_2')
    Arc2Model1 = MaverickArchitecture2()
    Arc2Model1.load_weights('weights_T5_withBiLSTM_TransformerNet_altOnly_model_4')
    Arc2Model2 = MaverickArchitecture2()
    Arc2Model2.load_weights('weights_T5_withBiLSTM_TransformerNet_altOnly_model_5')
    Arc2Model3 = MaverickArchitecture2()
    Arc2Model3.load_weights('weights_T5_withBiLSTM_TransformerNet_altOnly_model_7')
    Arc2Model4 = MaverickArchitecture2()
    Arc2Model4.load_weights('weights_T5_withBiLSTM_TransformerNet_altOnly_classWeights_1_2_3_model_1')
    Arc2Model5 = MaverickArchitecture2()
    Arc2Model5.load_weights('weights_T5_withBiLSTM_TransformerNet_altOnly_classWeights_1_2_7_model_1')

    # prep the data
    inputData=pandas.read_csv(inBase + '.annotated.txt',sep='\t',low_memory=False)
    inputData.loc[inputData['GDI']>2000,'GDI']=2000
    inputDataNonSeqInfo=inputData[['controls_AF','controls_nhomalt','pLI','pNull','pRec','mis_z','lof_z','CCR','GDI','pext','RVIS_ExAC_0.05','gerp']].copy(deep=True)
    inputDataNonSeqInfo.loc[inputDataNonSeqInfo['controls_AF'].isna(),'controls_AF']=0
    inputDataNonSeqInfo.loc[inputDataNonSeqInfo['controls_nhomalt'].isna(),'controls_nhomalt']=0
    inputDataNonSeqInfo.loc[inputDataNonSeqInfo['controls_nhomalt']>10,'controls_nhomalt']=10
    inputDataNonSeqInfo=inputDataNonSeqInfo.fillna(trainingDataNonSeqMedians)
    inputDataNonSeqInfo=np.asarray(inputDataNonSeqInfo.to_numpy()).astype(np.float32)
    # scale columns by QT
    inputDataNonSeqInfo=qt.transform(inputDataNonSeqInfo)
    inputData.loc[:,['controls_AF','controls_nhomalt','pLI','pNull','pRec','mis_z','lof_z','CCR','GDI','pext','RVIS_ExAC_0.05','gerp']]=inputDataNonSeqInfo

    data_generator=DataGenerator(np.arange(len(inputData)),np.ones(len(inputData)),dataFrameIn=inputData,tokenizer=tokenizer,T5Model=T5Model,batch_size=batchSize,shuffle=False)

    # set up the output collectors
    Arc1Model1Preds=inputData.loc[:,['hg38_chr','hg38_pos(1-based)','ref','alt']]
    Arc1Model1Preds['BenignScore']=0
    Arc1Model1Preds['DomScore']=0
    Arc1Model1Preds['RecScore']=0
    Arc1Model2Preds=Arc1Model1Preds.copy(deep=True)
    Arc1Model3Preds=Arc1Model1Preds.copy(deep=True)
    Arc2Model1Preds=Arc1Model1Preds.copy(deep=True)
    Arc2Model2Preds=Arc1Model1Preds.copy(deep=True)
    Arc2Model3Preds=Arc1Model1Preds.copy(deep=True)
    Arc2Model4Preds=Arc1Model1Preds.copy(deep=True)
    Arc2Model5Preds=Arc1Model1Preds.copy(deep=True)


    # score the test data
    for batchNum in range(int(np.ceil(len(inputData)/batchSize))):
        print('Starting batch number ' + str(batchNum), flush=True)
        thisBatch=data_generator[batchNum]
        thisBatchT5={'alt_cons':thisBatch[0]['alt_cons'],'alt_emb':thisBatch[0]['alt_emb'],'non_seq_info':thisBatch[0]['non_seq_info']}
        thisBatchDiff={'mm_orig_seq':thisBatch[0]['mm_orig_seq'],'mm_alt_seq':thisBatch[0]['alt_cons'],'non_seq_info':thisBatch[0]['non_seq_info']}
        Arc1Model1Preds.loc[(batchNum*batchSize):((batchNum*batchSize)+len(thisBatch[1])-1),['BenignScore','DomScore','RecScore']]=Arc1Model1.predict(thisBatchDiff,verbose=0)
        Arc1Model2Preds.loc[(batchNum*batchSize):((batchNum*batchSize)+len(thisBatch[1])-1),['BenignScore','DomScore','RecScore']]=Arc1Model2.predict(thisBatchDiff,verbose=0)
        Arc1Model3Preds.loc[(batchNum*batchSize):((batchNum*batchSize)+len(thisBatch[1])-1),['BenignScore','DomScore','RecScore']]=Arc1Model3.predict(thisBatchDiff,verbose=0)
        Arc2Model1Preds.loc[(batchNum*batchSize):((batchNum*batchSize)+len(thisBatch[1])-1),['BenignScore','DomScore','RecScore']]=Arc2Model1.predict(thisBatchT5,verbose=0)
        Arc2Model2Preds.loc[(batchNum*batchSize):((batchNum*batchSize)+len(thisBatch[1])-1),['BenignScore','DomScore','RecScore']]=Arc2Model2.predict(thisBatchT5,verbose=0)
        Arc2Model3Preds.loc[(batchNum*batchSize):((batchNum*batchSize)+len(thisBatch[1])-1),['BenignScore','DomScore','RecScore']]=Arc2Model3.predict(thisBatchT5,verbose=0)
        Arc2Model4Preds.loc[(batchNum*batchSize):((batchNum*batchSize)+len(thisBatch[1])-1),['BenignScore','DomScore','RecScore']]=Arc2Model4.predict(thisBatchT5,verbose=0)
        Arc2Model5Preds.loc[(batchNum*batchSize):((batchNum*batchSize)+len(thisBatch[1])-1),['BenignScore','DomScore','RecScore']]=Arc2Model5.predict(thisBatchT5,verbose=0)

    # save individual model results to file
    inputData.loc[:,['Arc1Model1_BenignScore','Arc1Model1_DomScore','Arc1Model1_RecScore']]=Arc1Model1Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    inputData.loc[:,['Arc1Model2_BenignScore','Arc1Model2_DomScore','Arc1Model2_RecScore']]=Arc1Model2Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    inputData.loc[:,['Arc1Model3_BenignScore','Arc1Model3_DomScore','Arc1Model3_RecScore']]=Arc1Model3Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    inputData.loc[:,['Arc2Model1_BenignScore','Arc2Model1_DomScore','Arc2Model1_RecScore']]=Arc2Model1Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    inputData.loc[:,['Arc2Model2_BenignScore','Arc2Model2_DomScore','Arc2Model2_RecScore']]=Arc2Model2Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    inputData.loc[:,['Arc2Model3_BenignScore','Arc2Model3_DomScore','Arc2Model3_RecScore']]=Arc2Model3Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    inputData.loc[:,['Arc2Model4_BenignScore','Arc2Model4_DomScore','Arc2Model4_RecScore']]=Arc2Model4Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    inputData.loc[:,['Arc2Model5_BenignScore','Arc2Model5_DomScore','Arc2Model5_RecScore']]=Arc2Model5Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()

    # ensemble results together
    y_pred1=Arc1Model1Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    y_pred2=Arc1Model2Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    y_pred3=Arc1Model3Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    y_pred4=Arc2Model1Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    y_pred5=Arc2Model2Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    y_pred6=Arc2Model3Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    y_pred7=Arc2Model4Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    y_pred8=Arc2Model5Preds.loc[:,['BenignScore','DomScore','RecScore']].to_numpy()
    y_pred=np.mean([y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7,y_pred8],axis=0)
    inputData.loc[:,['Maverick_BenignScore','Maverick_DomScore','Maverick_RecScore']]=y_pred
    inputData.to_csv(inBase + '.MaverickResults.txt',sep='\t',index=False)
    return


if __name__ == "__main__":
    main(sys.argv[1:])

