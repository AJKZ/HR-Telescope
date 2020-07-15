#include <iostream>
#include <fstream>
#include <stdint.h>
#include <vector>
#include <thread>
// #include <time.h>
//#include "GenericList.hpp" 

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

using namespace cv::xfeatures2d;
using namespace cv;
using namespace std;

// GenericList<cv::Mat frame> list;
// match = cv2.BFMatcher(cv2.NORM_HAMPING, crossCheck =True);

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;
double lifespan = 200;
float off_x;
float off_y;

void displayImage(cv::Mat source);
void stacking(cv::Mat& source);
void alignImages(Mat& im1, Mat& im2, Mat& im1Reg, Mat& h);
cv::Mat stacked_image;

int main(int argv, char** argc)
{
    Mat image, secImage, h;

    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        return -1;
    }

    while (1) {
        Mat frame;
        bool bSuccess = cam.read(frame); // read a new frame from video 

        //Breaking the while loop if the frames cannot be captured
        if (bSuccess == false)
        {
            cout << "Video camera is disconnected" << endl;
            cin.get(); //Wait for any key press
            break;
        }
        // wait till list is filled with 64 photo's
        // if(list.getSize() >= 64)
        // {
        // 	// waitkey(list.getSize == 0);
        // }	
        else
        {
            
            Mat& image = frame;
            Mat& secImage = frame;
            // image = imread(cam.read(frame);
            Mat newPhoto;
            alignImages(image, secImage, newPhoto, h);
            Mat grayScaleFrame;
            cvtColor(newPhoto, grayScaleFrame, COLOR_BGR2GRAY);
            string outFile("Alligned.jpeg");
            imwrite(outFile, newPhoto);
            imshow("Greyscale photo", newPhoto);
            cout << "homography ::" << h << endl;
            // list.push(image, off_x, off_y, lifespan);
        }
    }
}


void::displayImage(cv::Mat source)
{
    // split the source
    Mat splitArray[2] = { Mat::zeros(source.size(), source.type()), Mat::zeros(source.size(), source.type()) };
    split(source, splitArray);
    // take magnitude
    Mat dftMagnitude;
    magnitude(splitArray[0], splitArray[1], dftMagnitude);
    // add 1 to all values and take the log
    dftMagnitude += Scalar::all(1);
    log(dftMagnitude, dftMagnitude);
    // normalize matrix
    normalize(dftMagnitude, dftMagnitude, 0, 1, NORM_MINMAX);
    // display
    imshow("image", dftMagnitude);
    //waitKey();	
}

void alignImages(Mat& im1, Mat& im2, Mat& im1Reg, Mat& h)
//https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
{
    // Variables to store keypoints and descriptors
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    // Detect ORB features and compute descriptors.
    Mat im1Gray, im2Gray;
    cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
    cvtColor(im2, im2Gray, COLOR_BGR2GRAY);
    Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
    orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
    // Match features.
    std::vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, Mat());
    // Sort matches by score
    std::sort(matches.begin(), matches.end());
    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin() + numGoodMatches, matches.end());
    // Draw top matches
    Mat imMatches;
    drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
    imwrite("matches.jpg", imMatches);
    // Extract location of good matches
    std::vector<Point2f> points1, points2;

    for (size_t i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    // Find homography
    h = findHomography(points1, points2, RANSAC);
    // Use homography to warp image
    warpPerspective(im1, im1Reg, h, im2.size());

}
 