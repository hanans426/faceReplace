// Summary: 利用OpenCV的LBF算法进行人脸关键点检测
// Author:  Amusi
// Date:    2018-03-20
// Reference:
//		[1]Tutorial: https://www.learnopencv.com/facemark-facial-landmark-detection-using-opencv/
//		[2]Code: https://github.com/spmallick/learnopencv/tree/master/FacialLandmarkDetection

// Note: OpenCV3.4以及上支持Facemark
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"
#include <tuple>


using namespace std;
using namespace cv;
using namespace cv::face;

extern int drawLandmarks(vector<vector<Point2f>>, Mat);
extern int drawGlass(vector<vector<Point2f>>, Mat);
extern int drawBeard(vector<vector<Point2f>>, Mat);


int main(int argc,char** argv)
{
    // 加载人脸检测器（Face Detector）
    // [1]Haar Face Detector
   // CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
    // [2]LBP Face Detector
	CascadeClassifier faceDetector("/Users/gaohan/CLionProjects/faceReplace/include/lbpcascade_frontalface.xml");

    // 创建Facemark类的对象
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // 加载人脸检测器模型
    facemark->loadModel("/Users/gaohan/CLionProjects/faceReplace/include/lbfmodel.yaml");

    // 设置网络摄像头用来捕获视频
    VideoCapture cam(0);

    // 存储视频帧和灰度图的变量
    Mat frame, gray;

    // 读取帧
    while(cam.read(frame))
    {

        // 存储人脸矩形框的容器
      vector<Rect> faces;
        // 将视频帧转换至灰度图, 因为Face Detector的输入是灰度图
      cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 人脸检测
      faceDetector.detectMultiScale(gray, faces);

        // 人脸关键点的容器
      vector< vector<Point2f> > landmarks;

        // 运行人脸关键点检测器（landmark detector）
      bool success = facemark->fit(frame,faces,landmarks);
      
      if(success)
      {
          // 如果成功, 在视频帧上绘制
         //drawLandmarks(landmarks, frame);    //绘制关键点
         drawGlass(landmarks, frame);      //绘制眼镜
         //drawBeard(landmarks, frame);       //绘制胡子

      }

        // 显示结果
      imshow("Facial Landmark Detection", frame);

        // 如果按下ESC键, 则退出程序
      if (waitKey(1) == 27) break;
      
    }
    return 0;
}

int drawLandmarks(vector<vector<Point2f>> landmarks, Mat frame) {
    for(int i = 0; i < landmarks.size(); i++){
        // 自定义绘制人脸特征点函数, 可绘制人脸特征点形状/轮廓
        //drawLandmarks(frame, landmarks[i]);
        // OpenCV自带绘制人脸关键点函数: drawFacemarks
        drawFacemarks(frame, landmarks[i], Scalar(0, 0, 255));
    }
}

int drawGlass(vector<vector<Point2f>> landmarks, Mat frame) {
    for (int i = 0; i < landmarks.size(); i++) {
        auto v = landmarks[i];
        std::tuple<float, float> pos_left = make_tuple(v[0].x, v[36].y);
        std::tuple<float, float> pos_right = make_tuple(v[16].x, v[45].y);
        std::tuple<float, float> face_center = make_tuple(v[27].x, v[27].y);

        float width = std::get<0>(pos_right) - std::get<0>(pos_left);


        Mat glass = imread("/Users/gaohan/CLionProjects/faceReplace/resource/img/glass.png");

        try {

            float scale = glass.cols / width;
            float height = glass.rows / scale;
            Mat resizeMat;
            resize(glass, resizeMat, Size2f(width, height));
            Mat src_mask = Mat::zeros(resizeMat.cols, resizeMat.rows, resizeMat.type());

            Point2f p;
            p.x = std::get<0>(face_center);
            p.y = std::get<1>(face_center);

            bitwise_or(resizeMat, src_mask, src_mask);
            bitwise_not(src_mask, src_mask);
            seamlessClone(resizeMat, frame, src_mask, p, frame, MIXED_CLONE);

        } catch (exception) {
            cout << "some error happend " << endl;
        }


    }
}

int drawBeard (vector<vector<Point2f>> landmarks, Mat frame) {
    for(int i = 0; i < landmarks.size(); i++) {
        auto v = landmarks[i];
        std::tuple<float, float> pos_left = make_tuple(v[48].x, v[34].y);
        std::tuple<float, float> pos_right = make_tuple(v[54].x, v[34].y);
        std::tuple<float, float> face_center = make_tuple(v[33].x, v[33].y);

        float width = std::get<0>(pos_right) - std::get<0>(pos_left);


        Mat glass = imread("/Users/gaohan/CLionProjects/faceReplace/resource/img/beard.png");

        try {

            float scale = glass.cols / width;
            float height = glass.rows / scale;
            Mat resizeMat;
            resize(glass, resizeMat, Size2f(width, height));
            Mat src_mask = Mat::zeros(resizeMat.cols, resizeMat.rows, resizeMat.type());

            Point2f p;
            p.x = std::get<0>(face_center);
            p.y = std::get<1>(face_center);

            bitwise_or(resizeMat, src_mask, src_mask);
            bitwise_not(src_mask, src_mask);
            seamlessClone(resizeMat, frame, src_mask, p, frame, MIXED_CLONE);

        } catch (exception) {
            cout << "some error happend " << endl;
        }
    }

}