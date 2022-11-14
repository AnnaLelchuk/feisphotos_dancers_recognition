class Embeddings:
  def __init__(self, pictures):
    """ TBD """
    self.pictures = pictures
    pass

   def main(self):
      embeddings = []
      for pic in self.pictures:
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
    
    
