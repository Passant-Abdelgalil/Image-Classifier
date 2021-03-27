import sys
import argparse
from utilties import *

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Image path you wish to predict", type=str)
    parser.add_argument("model_path", help="Saved model path you wish to use in prediction", type=str)
    parser.add_argument("--top_k",help="Number of topmost predicted classes",type=int)
    parser.add_argument("--category_names",help="JSON file path for category names",type=str)
    
    args = parser.parse_args()
    
    image = read_image(args.image_path)
    model = load_model(args.model_path)
    top_k = 1

    if args.top_k:
        top_k = args.top_k
    
    image = process_image(image)
    probs, classes_labels = predict(image, model, top_k)
    probs = [round(probability,5) for probability in probs]
    
    if args.category_names:
        class_names = load_jsonify_classes(args.category_names)
        
        predicted_classes = [class_names[hashed] for hashed in classes_labels]
        print("The top %d predicted classes in descending order are:" %top_k, predicted_classes)
        print("with probabilities",probs)
    else:
        print("The top %d predicted classes in descending order are:" %top_k, classes_labels)
        print("with probabilities",probs)
        
        
if __name__=='__main__':
    main()
