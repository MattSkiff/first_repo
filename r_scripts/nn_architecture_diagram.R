library(DiagrammeR)
			
gr <-grViz("digraph flowchart {
      # node definitions with substituted label text
      node [fontname = Helvetica, shape = rectangle]        

      # edge definitions with the node IDs
      'Linear Combiner (Perceptron) (ANN)' -> 'Multiple Linear Combiner (One Layer Perceptron)';
			'Multiple Linear Combiner (One Layer Perceptron)'-> 'MLP (FFNN)'
			'MLP (FFNN)' -> 'RBF-NN';
			'Neocognitron' -> 'LeNet'-> 'AlexNet (DCNNs)';
			'AlexNet (DCNNs)' -> 'DeepFace'; 
			'AlexNet (DCNNs)' -> 'VGG'; 
			'AlexNet (DCNNs)' -> 'Inception Net'; 
			'AlexNet (DCNNs)' -> 'ResNet';
			'AlexNet (DCNNs)' -> 'R-CNN' -> 'Fast R-CNN';
			'AlexNet (DCNNs)' -> 'YOLO';
			'AlexNet (DCNNs)' -> 'SqueezeNet';
			'LeNet' -> 'GNN' -> 'GTN' -> 'GPT-2';
			'GTN' -> 'BERT';
			'MLP (FFNN)' -> 'ELM';
			'MLP (FFNN)' -> 'SNN';
			'MLP (FFNN)' -> 'SOM (CNNs)';
			'MLP (FFNN)' -> 'Autoencoders' -> 'DAE' -> 'VAE';
			'MLP (FFNN)' -> 'TDNN';
			'TDNN' -> 'RNNs' -> 'BRNN';
			'RNNs' -> 'LTSM';
			'SOM (CNNs)' -> 'Neocognitron'
			'MLP (FFNN)' -> 'Hopfield Nets' -> 'RNNs';
			'DBM' -> 'GSN'; 
			'MLP (FFNN)' -> 'Boltzmann Machine' -> 'RBM' -> 'DBM';
			'GSN' -> 'GAN' -> 'GauGAN';
			'GAN' -> 'StyleGAN';
      }
      ")

gr

# seem to have to use viewer for package - odd
# png(filename = "G:/My Drive/Dissertation/figures/nn_architectures.png",
# 		width = 800, height = 600, units = "px", 
# 		res = 120,
# 		bg = "white",  
# 		type = "cairo-png")
# gr
# 
# dev.off()

# RBF: https://apps.dtic.mil/docs/citations/ADA196234
# GauGAN: https://www.researchgate.net/publication/334714551_GauGAN_semantic_image_synthesis_with_spatially_adaptive_normalization