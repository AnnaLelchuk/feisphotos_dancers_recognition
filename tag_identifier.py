class TagIdentifier():
  """ Get tag numbers. NOT in use right now """

  def __init__(self):
    """ helper class to process our dataset """

    self.east_model = '/content/drive/MyDrive/east_model/frozen_east_text_detection.pb'
    self.net = cv.dnn.readNet(self.east_model)

    # needed for our model 
    self.outNames = []
    self.outNames.append("feature_fusion/Conv_7/Sigmoid")
    self.outNames.append("feature_fusion/concat_3")

  def identify_tag(self, src_img, show=False, conf_thr=None, nms_conf_thr=None):
    """" 
      main() wrapper method around other methods necessary to get the text. 
      main configurables - the thresholds and can use `show`=True to display 
      the boxes found.
    """
    
    # 1. configure thresholds.

    if conf_thr is None:
      raise("Must pass conf_thr")
    if nms_conf_thr is None:
      raise("Must pass nms_conf_thr")

    # CONFIDENCE TRESHOLD: increasing decreases boxes found, decreasing more boxes found.
    CONF_THRESH = conf_thr       
    # NON MAXIMUM SUPRESSION: increasing leaves more indicies, decreasing less boxes
    NMS_CONF_THRESH = nms_conf_thr   

    # 2. get blob
    blob = self.a1_create_blob(src_img)
    # 3. get scores + geometry
    scores, geometry = self.a2_get_box_scores(blob)
    # 4. filter out the relevant boxes 
    boxes, confidences = self.a3_decode_scores(scores, geometry, conf_thr=CONF_THRESH)   
    # 5. nms filtering and such, matches are boxed areas
    matches, indicies = self.a4_return_matches(src_img, boxes, confidences, show=show, conf_thr=CONF_THRESH, nms_conf_thr=NMS_CONF_THRESH)
    # 6. identify the boxes' text and return those
    text_results = self.a5_to_text(matches)

    return [matches, text_results]

  def a1_create_blob(self, src_img):
    """ input an image: get blob back for further cv2 + NN processing """ 

    # This needs to be the closest size to the original image passed in, as a multiple of 32 
    # due to the way the NN is designed and trained on.  used for the nn and to maintain ratios, factor of 32
    # TODO - reshape to the closest one of the input image vs this static size. 
    # However, From a quick scan it seems  like most of the iamges are of the same size and the below should work for image ratios:
    #    image shapes are pretty uniform at (2456, 36963, 3) 

    inpWidth = 2_432
    inpHeight = 3_680

    # actual
    width_ = src_img.shape[1]
    height_ = src_img.shape[0]

    # resize scaling factor since nn expects certain sizes and our image is of a certain size.
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)

    # do we want to scale the feature pixel themselves? 1.0 is "no" - just multiply by 1. 
    # could do 1/255 for example to limit the feature range
    pixel_feature_scaling = 1.0
    mean_scaling = (123.68, 116.78, 103.94)

    blob = cv.dnn.blobFromImage(src_img, pixel_feature_scaling, (inpWidth, inpHeight), mean_scaling, True, False)
    return blob

  def a2_get_box_scores(self, blob):
    """ use NN to scan the blb image for ID tags """

    # 1. if we care about metrics of inference time taken see below.
    # 2. it's possible we may need to move the net initializations to here vs __init__? unclear. try using __init__ first.
    # 
    #
    # t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #
    # self.net = cv.dnn.readNet(east)
  
    self.net.setInput(blob)
    outs = self.net.forward(self.outNames)
    scores, geometry = outs[0], outs[1]
    return [scores, geometry]

  def a3_decode_scores(self, scores, geometry, conf_thr=None):    
    """ 
      use scores and geometry to return detections of boxes and confidences of these.
      shamelessly taken from https://github.com/opencv/opencv/blob/4.x/samples/dnn/text_detection.py 
    """
    if conf_thr is None:
      raise("Must pass conf_thr")

    detections = []
    confidences = []
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):
        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]
            # If score is lower than threshold score, move to next x
            if(score < conf_thr):
                continue
            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])
            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

  def a4_return_matches(self, src_img, boxes, confidences, show=False, conf_thr=None, nms_conf_thr=None):

    """
    input:
        [boxes, confidences] = self.a3_decode(scores, geometry, conf_thr)
        as well as confidence threshold & nms_conf_thr ( non maximum suppression ).

      ...show boxes that match our confidence levels. 

    """
    if conf_thr is None:
      raise("Must pass conf_thr")
    if nms_conf_thr is None:
      raise("Must pass nms_conf_thr")

    # this returns a few indicies that map out the box coordinates and confidence levels based on our thresholds etc.
    indices = cv.dnn.NMSBoxesRotated(boxes, confidences, conf_thr, nms_conf_thr)
    
    # # if verbose
    #
    # display(len(boxes), len(confidences))
    # print("Example boxes/confidence:", boxes[1], confidences[1])
    # print("Num box indicies found:", len(indices))

    # TODO/XXX - for now, draw out these potential matches in a bunch and see what we get.
    # TODO - tweak confidence scores thresholds and such, find and return only one cropped image \
    # of the given tag or "-1" for could not identify any number tag. 
    # 
    # also consider the other text identification that may occur, and sanely return JUST THE NUMBER TAG or -1. 

    matches = []
    for idx, i in enumerate(indices):

        # print(i, boxes[i], confidences[i])
        # display(f"i: {i}, box: {boxes[i]}, conf:{confidences[i]}")
        
        x1, y1 = boxes[i][0]
        width, height = boxes[i][1]
        x1, y1, width, height = int(x1), int(y1), int(width), int(height)
        x2, y2 = x1 + width, y1 + height
        x1, y1 = x1-width, y1-height

        # extract the area we want to focus on:
        tag_area = src_img[y1:y2, x1:x2]
        matches.append(tag_area)

        if show:
          plt.title(f"i: {i}, box: {boxes[i]}, conf:{confidences[i]}") 
          plt.imshow(tag_area)
          plt.show()
    
    return matches, indices 

  def a5_to_text(self, matches):
    """ return text matches from the box matches we got earlier stages """

    texts = []
    for mdx, match in enumerate(matches):
      
      print(f"Text detection for match {mdx}:")
      thresh = cv.threshold(match, 100, 255, cv.THRESH_BINARY)[1] # convert to binary first ( Why? Nissim? )

      data = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6')        
      texts.append(data)

    return texts
