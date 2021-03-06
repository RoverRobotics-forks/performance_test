// Copyright 2017 Apex.AI, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMMUNICATION_ABSTRACTIONS__ROS2_WAITSET_COMMUNICATOR_HPP_
#define COMMUNICATION_ABSTRACTIONS__ROS2_WAITSET_COMMUNICATOR_HPP_

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <atomic>

#include "../experiment_configuration/topics.hpp"
#include "../experiment_configuration/qos_abstraction.hpp"

#include "ros2_communicator.hpp"
#include "resource_manager.hpp"

namespace performance_test
{

/// Communication plugin for ROS 2 using waitsets for the subscription side.
template<class Topic>
class ROS2WaitsetCommunicator : public ROS2Communicator<Topic>
{
public:
  /// The data type to publish and subscribe to.
  using DataType = typename ROS2Communicator<Topic>::DataType;

  /// Constructor which takes a reference \param lock to the lock to use.
  explicit ROS2WaitsetCommunicator(SpinLock & lock)
  : ROS2Communicator<Topic>(lock),
    m_polling_subscription(nullptr) {}

  /// Reads received data from ROS 2 using waitsets
  void update_subscription() override
  {
    if (!m_polling_subscription) {
      m_polling_subscription = this->m_node->template create_polling_subscription<DataType>(
        Topic::topic_name() + this->m_ec.sub_topic_postfix(), this->m_ROS2QOSAdapter);
      m_waitset = std::make_unique<rclcpp::Waitset<>>(m_polling_subscription);
    }
    try {
      const auto wait_ret = m_waitset->wait(std::chrono::milliseconds(100));
      this->lock();
      if (wait_ret.any()) {
        auto received_sample = m_polling_subscription->take();
        if (received_sample) {
          this->template callback(received_sample.data());
        }
      }
    } catch (const rclcpp::TimeoutError &) {
      std::cerr << "Waitset timed out without receiving a sample." << std::endl;
    }
    this->unlock();
  }

private:
  std::shared_ptr<::rclcpp::PollingSubscription<DataType>> m_polling_subscription;
  std::unique_ptr<rclcpp::Waitset<>> m_waitset;
};

}  // namespace performance_test

#endif  // COMMUNICATION_ABSTRACTIONS__ROS2_WAITSET_COMMUNICATOR_HPP_
