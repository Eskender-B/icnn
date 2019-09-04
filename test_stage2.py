from bg_modulate import Modulator
from end2end import *


def main():
  
  with torch.no_grad():
    for batch in test_loader:
      images, labels, indexs = batch['image'].to(device), batch['labels'].to(device), batch['index']


      ## Find locations from ground truth
      orig_labels = F.normalize(labels, 1)
      centroids = calculate_centroids(orig_labels)

      ## Extract patches from face from their location given in centroids
      # Get also shift-scaled centroids, offsets and shapes 
      parts, centroids, orig, offsets, shapes = extract_parts(indexs, centroids, unresized_dataset)

      ## Prepare batches for facial parts
      batches = prepare_batches(parts)
     
      ## Get prediction
      pred_labels = {}
      for name in batches:
        pred_labels[name] = F.one_hot(modulators[name](models[name](batches[name]['image'])).argmax(dim=1), models[name].L).transpose(3,1).transpose(2,3)

      ## Update F1-measure stat for this batch
      calculate_F1(batches, pred_labels)

      ## Rearrange patch results onto original image
      ground_result, pred_result = combine_results(pred_labels, orig, centroids)

      ## Save results
      save_results(ground_result, pred_result, indexs, offsets, shapes)
      print("Processed %d images"%args.batch_size)

  ## Show stats
  show_F1()

if __name__ == '__main__':
  main()