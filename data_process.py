import os
from pickle import FALSE
import numpy as np
import pretty_midi
import shutil 
import music21
from music21 import note, chord, duration, pitch
import pypianoroll
print('___ALL INSTALLED___')

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

#Split Multitrack midi into single track midi
def separate_tracks():
    pathname = os.getcwd()
    for file in os.listdir('KPOP data files'):
        filewithpath = "KPOP data files/" + file
        #print(filewithpath)
        data = pretty_midi.PrettyMIDI(filewithpath)
        name = file.split('.')[0]
        for index in range(len(data.instruments)):
            #list of notes to append
            note = data.instruments[index].notes

            #empty output midi with grand piano instrument
            output = pretty_midi.PrettyMIDI()
            output.instruments.append(pretty_midi.Instrument(program = 1))
            output.instruments[0].notes = note
            filename = name + str(index) + ".mid"
            output.write(filename)
            shutil.move(filename, pathname + '/Split_KPOP')

#Split_KPOP file has all the single track midi file
#We make more processing to make it feedable to architecture

#retrieving a list of info only for notes, rests and chords
def screen_relevant(midi_notes):
    screenedlist = []
    for sound in midi_notes:
        if isinstance(sound, note.Note) or isinstance(sound, note.Rest) or isinstance(sound, chord.Chord):
                screenedlist.append(sound) 
    return screenedlist

#gets rid of the starting resting measures, these are irrelevant to style
def trim_start(midi_notes):
    trimmedlist = []
    start = False
    for sound in midi_notes:
        if start:
            trimmedlist.append(sound)
        else:
            if isinstance(sound, note.Note):
                trimmedlist.append(sound)
                start = True
    return trimmedlist

#adjusts the timestamps of each notes' duration relative to the length of 16th notes duration 
def transform16th(midi_notes):
    resampledlist = []
    sixteenlength = duration.Duration(0.25)
    for sound in midi_notes:
        #cutting the note duration into 4 pieces
        adjustedlen = int(sound.duration.quarterLength // 0.25) + 1
        if isinstance(sound, note.Note):
            for i in range(adjustedlen):
                resizedNote = note.Note(pitchName=sound.pitch, duration = sixteenlength)
                resampledlist.append(resizedNote)
        elif isinstance(sound, chord.Chord):
            for i in range(adjustedlen):
                resizedChord = chord.Chord(sound.notes, duration=sixteenlength) 
                resampledlist.append(resizedChord)
        elif isinstance(sound, note.Rest):
                resizedRest = note.Rest(duration=sixteenlength)
                resampledlist.append(resizedRest)
    return resampledlist

#check if currentNote is within range from c1-c8
def check_pitch(currentNote):
    c1 = pitch.Pitch('C1')
    c8 = pitch.Pitch('C8')
    # print(c1)
    # print(c8)
    # print(type(currentNote))
    # print(type(c1))
    # print(type(c8))
    if currentNote > c8 or currentNote < c1:
        return FALSE
    else:
        return True

#prepare a passable numpy array with batchsize [64, 84, 1]
def convert_array():
    pathname = os.getcwd()
    for file in os.listdir('Split_KPOP'):
        #make a stream of midifile
        tmp_path =  'Split_KPOP/' + file
        songname =  tmp_path.split('.')[0]
        midi_stream = music21.converter.parse(tmp_path)
        #print(type(midi_stream))
        #pass stream to make instrument parts(piano)
        midi_notes = music21.instrument.partitionByInstrument(midi_stream).parts[0].recurse()
        #midi_notes.show('text')
        #trim the track so that empty starting bars are taken out
        trimmed_notes = trim_start(screen_relevant(midi_notes))
        #print(trimmed_notes[0].duration.quarterLength)
        #print(trimmed_notes[0].pitches)
        print("___successfully trimmed!___")
        #print(len(trimmed_notes))
        #create a narray with only notes, rests and chords
        #Note that each note value will have a duration in relation to 16th note length and we will create samples of length 4 measures which is 64 notes
        resized_notes = transform16th(trimmed_notes)
        #Using the resized_notes, we will make a npy file every 64 instances
        sampleCount = 0
        currentnpy = 0
        print(len(resized_notes))
        #while condition omits the last sequences of notes, chords, and rests in order to produce array correctly
        while (sampleCount*64 + 64) < len(resized_notes):
            matrix = np.zeros(shape=(64, 84, 1), dtype=np.bool)
            for count in range(64):
                sound = resized_notes[sampleCount*64 + count]
                if isinstance(sound, note.Rest):
                    continue
                if isinstance(sound, note.Note):
                    currentNote = sound.pitch
                    if check_pitch(currentNote):
                        pitch_index = int(currentNote.ps) - 25
                        matrix[count, pitch_index, 0] = True
                elif isinstance(sound, chord.Chord):
                    for eachNote in sound.pitches:
                        if check_pitch(eachNote):
                            pitch_index = int(eachNote.ps) - 25
                            matrix[count, pitch_index, 0] = True
            filename = songname + '#'+ str(currentnpy) + '.npy'
            np.save(file=filename, arr=matrix)
            shutil.move(filename, pathname + '/ready_datafiles')
            sampleCount = sampleCount + 1
            currentnpy = currentnpy + 1
            print("___produced " + str(sampleCount) + "th file!___")

        print("___successfully produced npy for " + songname + ". Produced " + str(currentnpy) + " files!___")

        
    
    # outputpath = '/Split KPOP'
    # songname = 'Panorama'
    # counter = 0
    # #for midi_file_name in os.listdir():
    #     #print(midi_file_name)
    # for index in data.instruments:

    #     # filename = songname + '_' +  str(counter) + '.mid'
    #     # completeName = os.path.join(outputpath, filename)
    #     # file = open(completeName, "w")

    #     outfile = pretty_midi.PrettyMIDI()
    #     instrument = pretty_midi.Instrument(program = 1)
    #     notes = index.notes
    #     instrument.notes.append(notes)
    #     outfile.instruments.append(instrument)
    #     outfile.write(songname + str(counter) + ".mid") 
    #     counter = counter + 1
    #     #print(notes)
    # #print(data.instruments)
    # #print(data.estimate_tempo())
    # print(pretty_midi.PrettyMIDI("Panorama1.mid").get_piano_roll())


#separate_tracks()
#convert_array()
#check_pitch()