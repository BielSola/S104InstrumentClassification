import dataset_creation
import os

def main():
    """
    Main function to run the script
    """

    # Load the dataset
   
    
    dt_path = 'G:/.shortcut-targets-by-id/17yphSXB2IgKWLJF-VDo9xJDWWM2e6mkH/S104/dataset'
    saraga = dataset_creation.create_dataset(dt_path)

    track = saraga.choice_track()
    # Get the track ID from the user
    track_id = track.track_id
    # Get metadata for the track
    #metadata = dataset_creation.get_metadata(track_id)

    mixed_array = dataset_creation.load_mixed_audio(track_id)

    dataset_creation.play_audio(mixed_array)
    
    """
    # Get performer for the track
    performer = dataset_creation.get_performer(track_id, saraga)
    print(f"Performer: {performer}")

    # Get performance name for the track
    performance = dataset_creation.get_performance(track_id, saraga)
    print(f"Performance: {performance}")

    # Get raga name for the track
    raga = dataset_creation.get_raga(track_id, saraga)
    print(f"Raga: {raga}")
    """

if __name__ == "__main__":
    main()