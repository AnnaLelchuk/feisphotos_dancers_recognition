class Embeddings:
  def __init__(self, picture_paths):
    """ TBD """
    self.picture_paths = picture_paths
    pass

   def main(self):
      embeddings = []
      for pic in self.picture_paths:
        pass # returnb collection of ace/body embeddings
      return embeddings
      
   def face_embedding(self, pic):
      """ return VGGFace representation """
      # 
      # crop the face MTCNN
      # pass the input the face to VGGFace to get embeddings
      # return VGGFaceEmbed

      
   def body_embedding(self, pic):
      """ return VGG16 representation  """
      # crop the body YOLO
      # pass the input the face to VGG16 to get embeddings
      # return VGGFace
    
   def body_embedding_simease(self, pic):
      """ return Simease representation  """
      pass # TBD
    
    
##############################
# e = Embeddings(["/content/1.jpg", "/content/2.jpg", "/content/3.jpg"])
# # 1.jpg -> [1, 1]
# # 2.jpg -> [2, 2]
# assert(e.main(), [[1,1], [2,2]])
