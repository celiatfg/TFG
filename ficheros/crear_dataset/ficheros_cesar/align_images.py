# Registra fotos en un stack; las registra de 2 en dos
	# Input: stack de fotos
	# Output: stack de fotos registradas y lista de transformaciones para cada par de fotos

# import the necessary packages
import numpy as np
import imutils
import cv2
import imageio


# registra dos fotos; devuelve imagen registrada y transformación
#homography perspective H es 3x3 con 8 parámetros
# https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/

def align_2images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # use ORB to detect keypoints and extract (binary) local
	# invariant features
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	# match the features
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)
    # sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)
	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]
	# check to see if we should visualize the matched keypoints
	if debug:
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
			matches, None)
		matchedVis = imutils.resize(matchedVis, width=1000)
		cv2.imshow("Matched Keypoints", matchedVis)
		cv2.waitKey(0)

	# allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt
    # compute the homography matrix between the two sets of matched
	# points
	homography, mask = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	# use the homography matrix to align the images
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, homography, (w, h))
	# return the aligned image
	return aligned, homography


# Input: stack con fotos a alinear + hiperparámetros
# Output: fotos alineadas y matrices de transformación
def align_images(stack, maxFeatures=500, keepPercent=0.2, debug=False):
	stack_reg = stack.copy()
	lista_transf = np.zeros((stack.shape[0],3,3)) #stack de matrices de transformación
	lista_transf[0]=np.identity(3) #primera foto ya registrada
	for i in range(0,stack.shape[0]-1): #primera foto referencia, ya registrada
		template = stack[i]
		image_2 = stack[i+1]
		image_2_reg, H = align_2images(image_2, template, maxFeatures, keepPercent, debug)
		#imageio.imwrite("G:/fotografia/apps/circumpolar/jpegs/prueba_python/5fotos/registra/"+str(i+1)+".jpg", image_2_reg)
		stack_reg[i+1] = image_2_reg
		lista_transf[i+1] = H
	return stack_reg, lista_transf





