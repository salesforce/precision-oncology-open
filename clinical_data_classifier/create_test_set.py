"""Script to create a test set for oncologists
"""
from rtog_helper import RTOG

def generate_human_test_set(rtog, size=50, seed=9, field_to_balance='disease_free_survival'):
    # Generate balanced test set
    rtog_test_set_answers = rtog.generate_test_set(size=size, seed=seed, field_to_balance=field_to_balance)
    rtog_test_set = rtog_test_set_answers.clear_columns(columns=rtog_test_set_answers.endpoint_fields())

    # Drop any endpoint that have 'years' in the name. Humans won't be able to estimate this, after all.
    def drop_years(rtog):
        for c in rtog.df.columns:
            if 'years' in c:
                rtog = rtog.drop(columns=[c])
        return rtog

#   rtog_test_set_answers = drop_years(rtog_test_set_answers)
#   rtog_test_set = drop_years(rtog_test_set)
    return rtog_test_set_answers, rtog_test_set

if __name__ == "__main__":
    # Note: these settings work well for 9202. Adjust with caution.
    study_number='9202'
    size=50
    seed=9
    field_to_balance='disease_free_survival'

    print("Generating oncologist test for study {}. \nSize={}, seed={}, field_to_balance={}".format(
        study_number, size, seed, field_to_balance))
    rtog = RTOG(RTOG.gcp_baseline_paths[study_number])
    rtog.set_study_number(study_number)
    answers, questions = generate_human_test_set(rtog)
    answers_file = "./rtog_{}_human_test_answers.csv".format(answers.study_number)
    questions_file = "./rtog_{}_human_test_questions.csv".format(questions.study_number)
    answers.to_csv(answers_file)
    questions.to_csv(questions_file)
    print("Writing: {}".format(answers_file))
    print("Writing: {}".format(questions_file))

    print("Endpoint Statistics")
    for field in rtog.endpoint_fields():
        if 'years' in field:
            continue
        print(answers.df[[field]].value_counts())
