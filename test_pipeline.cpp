// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <vector>
#include <string>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/abstract_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/pinhole_camera.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <iostream>
#include <emscripten/bind.h>
#include "test_utils.h"

namespace svo {

  class BenchmarkNode
  {
      vk::AbstractCamera* cam_;
      svo::FrameHandlerMono* vo_;

    public:
      BenchmarkNode();
      ~BenchmarkNode();
      void runFromCamera(cv::Mat img, int img_id);
  };

  BenchmarkNode::BenchmarkNode()
  {
    cam_ = new vk::PinholeCamera(752, 480, 315.5, 315.5, 376.0, 240.0);
    vo_ = new svo::FrameHandlerMono(cam_);
    vo_->start();
  }

  BenchmarkNode::~BenchmarkNode()
  {
    delete vo_;
    delete cam_;
  }

  void BenchmarkNode::runFromCamera(cv::Mat img, int img_id)
  {
    cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
    assert(!img.empty());
    // process frame
    vo_->addImage(img, 0.01*img_id);

    SE3 T_world_from_vision_;
    SE3 T_world_from_cam;
    Vector3d p; 

    T_world_from_vision_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
    T_world_from_cam = SE3(T_world_from_vision_*vo_->lastFrame()->T_f_w_.inverse());
    p = Vector3d(vo_->lastFrame()->T_f_w_.translation());

    // display tracking quality
    if(vo_->lastFrame() != NULL)
    {
  //          std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
            std::cout << "#Features: " << vo_->lastNumObservations() << " \t";
  //                    << "Proc. Time: " << vo_->lastProcessingTime()*1000 << "ms \t"
        std::cout << "X: " << p[0] << " \t"
                  << "Y: " << p[1] << " \t"
                  << "Z: " << p[2] << "\n";

        // access the pose of the camera via vo_->lastFrame()->T_f_w_.
    }
  }
}

EMSCRIPTEN_BINDINGS(BenchmarkNode) {
    emscripten::class_<svo::BenchmarkNode>("BenchmarkNode")
      .constructor<>()
      .function("runFromCamera", &svo::BenchmarkNode::runFromCamera);
}
