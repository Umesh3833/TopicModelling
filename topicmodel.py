import little_mallet_wrapper
import seaborn
import glob
from pathlib import Path
import os
os.environ["MALLET_HOME"]= "C:\mallet"
path_to_mallet = "C:\mallet\bin\mallet"
directory = "C:\mallet\sample-data\web\data"
files = glob.glob(f"{directory}/*.txt")

training_data = []
for file in files:
    text = open(file, encoding='utf-8').read()
    processed_text = little_mallet_wrapper.process_string(text, numbers='remove')
    training_data.append(processed_text)

original_texts = []
for file in files:
    text = open(file, encoding='utf-8').read()
    original_texts.append(text)

obit_titles = [Path(file).stem for file in files]
print(obit_titles)
little_mallet_wrapper.print_dataset_stats(training_data)
num_topics = 5
training_data = training_data
#Change to your desired output directory
output_directory_path = r"C:/Users/Admin/Desktop/Nivi/topicmodel_output"

#No need to change anything below here
Path(f"{output_directory_path}").mkdir(parents=True, exist_ok=True)

path_to_training_data           = f"{output_directory_path}/training.txt"
path_to_formatted_training_data = f"{output_directory_path}/mallet.training"
path_to_model                   = f"{output_directory_path}/mallet.model.{str(num_topics)}"
path_to_topic_keys              = f"{output_directory_path}/mallet.topic_keys.{str(num_topics)}"
path_to_topic_distributions     = f"{output_directory_path}/mallet.topic_distributions.{str(num_topics)}"

little_mallet_wrapper.quick_train_topic_model(path_to_mallet,
                                             output_directory_path,
                                             num_topics,
                                             training_data)

topics = little_mallet_wrapper.load_topic_keys(path_to_topic_keys)
print(topics)
for topic_number, topic in enumerate(topics):
    print(f"---------------Topic {topic_number}-------------------\n\n{topic}\n")

    topic_distributions = little_mallet_wrapper.load_topic_distributions(path_to_topic_distributions)

obituary_to_check = "Data Commodity"

obit_number = obit_titles.index(obituary_to_check)

print(f"Topic Distributions for {obit_titles[obit_number]}\n")
for topic_number, (topic, topic_distribution) in enumerate(zip(topics, topic_distributions[obit_number])):
    print(f"Topic {topic_number} {topic[:6]} \nProbability: {round(topic_distribution, 6)}\n")
target_labels = obit_titles
little_mallet_wrapper.plot_categories_by_topics_heatmap(obit_titles,
                                      topic_distributions,
                                      topics, 
                                      output_directory_path + '/categories_by_topics.pdf',
                                      target_labels=target_labels,
                                      dim= (12, 15)
                                     )
training_data_obit_titles = dict(zip(training_data, obit_titles))
training_data_original_text = dict(zip(training_data, original_texts))

def display_top_titles_per_topic(topic_number=0, number_of_documents=5):
    
    print(f"Topic {topic_number}\n\n{topics[topic_number]}\n")

    for probability, document in little_mallet_wrapper.get_top_docs(training_data, topic_distributions, topic_number, n=number_of_documents):
        print(round(probability, 4), training_data_obit_titles[document] + "\n")
    return
display_top_titles_per_topic(topic_number=0, number_of_documents=5)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
for topic in topics:
    topic_words = topic
    wordcloud = WordCloud(width=800, height=800, random_state=21, max_font_size=110).generate(topic_words)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    plt.show()

#plot word cloud of display_top_titles_per_topic(topic_number=0, number_of_documents=5)
