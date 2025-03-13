import pandas as pd


def main(bbox_file, score_file, output_file):
    bbox_df = pd.read_csv(bbox_file)
    score_df = pd.read_csv(score_file)
    
    bbox_df.drop(columns=["finger"], inplace=True)
    score_df.drop(columns=["finger"], inplace=True)

    bbox_df.rename(columns={"Unnamed: 0": "joint_id"}, inplace=True)
    bbox_df["joint_id"] = bbox_df["joint_id"].apply(lambda x: x % 42)
    print(bbox_df)
    print(score_df)

    merged_df = pd.merge(
        score_df,
        bbox_df,
        left_on=['joint_id', 'patient_id', 'hand', 'joint'],
        right_on=['joint_id', 'patient_id', 'hand', 'joint'],
        how='inner'
    )
    
    merged_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main('C:\\Users\\User\\PycharmProjects\\ARISE\\dataset\\bboxes.csv','C:\\Users\\User\\PycharmProjects\\ARISE\\dataset\\scores.csv','C:\\Users\\User\\PycharmProjects\\ARISE\\dataset\\merges_box_scores.csv')
    