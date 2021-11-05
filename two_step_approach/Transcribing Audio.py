#Function to get Transcripts

def get_transcripts(source,destination):

    files = os.listdir(source)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #Importing Pretrained Models
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
    model = model.to(device)

    myfile = open(destination+"Transcripts.txt","w")

    for i in range(len(files)):
        
        print("Wav File "+str(i+1)+"/"+str(len(files)))
        
        #load any audio file of your choice
        speech, rate = librosa.load(source+files[i],sr=16000)
        input_values = processor(speech, return_tensors = 'pt').input_values.to(device)

        #Store logits (non-normalized predictions)
        logits = model(input_values).logits

        #Store predicted id's
        predicted_ids = torch.argmax(logits, dim =-1)

        #decode the audio to generate text
        transcriptions = processor.decode(predicted_ids[0])
        
        myfile.write(files[i]+";"+str(transcriptions))
        myfile.write('\n')
        
        del input_values, logits, predicted_ids, transcriptions, speech, rate

    myfile.close()
