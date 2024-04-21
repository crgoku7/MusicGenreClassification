import numpy as np
import librosa
import pickle
import streamlit as st
import soundfile as sf
import pandas as pd
import os

custom_css = """
    <style>
    .custom-link:hover {
        font-weight: bold;
    }
    </style>
"""

loaded_model = pickle.load(open('trained_model.sav', 'rb'))
loaded_pca = pickle.load(open('pca.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))


def music_transform(path):
    if ".mp3" in path:
        y, sr = sf.read(path)

        if len(y.shape) > 1:
            y = librosa.to_mono(y.T)

    y, sr = librosa.load(path)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)
    rms = librosa.feature.rms(y=y)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo = librosa.feature.tempo(y=y, sr=sr)

    data_array = []
    features = [chromagram, rms, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate,
                y_harmonic, y_percussive]

    for i in features:
        data_array.append(np.mean(i))
        data_array.append(np.var(i))

    data_array.append(tempo)

    for i in range(20):
        data_array.append(mfccs_mean[i])
        data_array.append(mfccs_var[i])

    music_data = pd.DataFrame(np.array(data_array, dtype="object").reshape(1, 57),
                              columns=['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
                                       'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean',
                                       'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var',
                                       'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean',
                                       'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo', 'mfcc1_mean',
                                       'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean',
                                       'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean',
                                       'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean',
                                       'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var',
                                       'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean',
                                       'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var',
                                       'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean',
                                       'mfcc20_var'])
    scaled_data_music = loaded_scaler.transform(music_data)

    scaled_dataframe = pd.DataFrame(scaled_data_music, columns=music_data.columns)

    musicpca_components = loaded_pca.transform(scaled_dataframe)

    music_pca = pd.DataFrame(musicpca_components)
    return music_pca


def prediction(path):
    transformed_music = music_transform(path)
    return loaded_model.predict(transformed_music)


def main():
    st.title("Music Genre Classifier")

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    genre = ''

    if uploaded_file is not None:
        # Create the "temp_files" directory if it doesn't exist
        os.makedirs("temp_files", exist_ok=True)

        # Save the uploaded file to a known location
        with open(os.path.join("temp_files", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Obtain the absolute path of the saved file
        file_path = os.path.abspath(os.path.join("temp_files", uploaded_file.name))

        if st.button("Predict"):
            genre = prediction(file_path)

    st.success(str(genre)[2:-2].capitalize())

    st.sidebar.title("Creators")
    st.markdown(custom_css, unsafe_allow_html=True)  # Inject custom CSS
    st.sidebar.write(
        '<a class="custom-link" style="color:#888888; text-decoration:none;" href="https://github.com/deepak5512">Deepak Bhatter</a>',
        unsafe_allow_html=True)
    st.sidebar.write(
        '<a class="custom-link" style="color:#888888; text-decoration:none;" href="https://github.com/Prajjwal-dixit">Prajjwal Dixit</a>',
        unsafe_allow_html=True)
    st.sidebar.write(
        '<a class="custom-link" style="color:#888888; text-decoration:none;" href="https://github.com/crgoku7">Tushar Bhatt</a>',
        unsafe_allow_html=True)
    st.sidebar.write(
        '<a class="custom-link" style="color:#888888; text-decoration:none;" href="https://github.com/RahulSharma6969">Rahul Sharma</a>',
        unsafe_allow_html=True)
    st.sidebar.write(
        '<a class="custom-link" style="color:#888888; text-decoration:none;" href="https://github.com/rhythmp17">Rhythm Patni</a>',
        unsafe_allow_html=True)
    st.sidebar.write(
        '<a class="custom-link" style="color:#888888; text-decoration:none;" href="https://github.com/RaphaelO07">Mayank Agrawal</a>',
        unsafe_allow_html=True)


if __name__ == '__main__':
    main()