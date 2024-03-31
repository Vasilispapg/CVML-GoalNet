import csv

def getAnnotations(annotation_path, videoID):
    annotations = []
    with open(annotation_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if row[0] == videoID:
                annotations.append(row[2].strip().split(','))
    return annotations