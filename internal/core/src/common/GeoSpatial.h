// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License
#pragma once

#include <gdal.h>
#include <ogr_geometry.h>
#include <string>
#include <string_view>
#include "log/Log.h"

namespace milvus {

class GeoSpatial {
 public:
    GeoSpatial() = default;

    // all ctr assume that wkb data and wkt string is valid
    explicit GeoSpatial(const std::string& wkt) {
        OGRGeometryFactory::createFromWkt(wkt.c_str(), nullptr, &geometry_);
        if (!geometry_) {
            LOG_WARN("[TH_DEBUG]ctr geospatial failed from wkt");
        } else {
            to_wkb_internal();
            to_wkt_internal();
        }
    }

    explicit GeoSpatial(const void* wkb, size_t size) {
        OGRGeometryFactory::createFromWkb(wkb, nullptr, &geometry_, size);
        if (!geometry_) {
            LOG_WARN("[TH_DEBUG]ctr geospatial failed from wkb");
        } else {
            to_wkb_internal();
            to_wkt_internal();
        }
    }

    GeoSpatial(const GeoSpatial& other) {
        if (other.IsValid()) {
            this->geometry_ = other.geometry_->clone();
            this->to_wkb_internal();
            this->to_wkt_internal();
        }
    }

    GeoSpatial(GeoSpatial&& other) noexcept
        : wkb_data_ro_(std::move(other.wkb_data_ro_)),
          wkb_data_wr_(std::move(other.wkb_data_wr_)),
          wkt_data_(std::move(other.wkt_data_)),
          geometry_(std::move(other.geometry_)) {
    }

    GeoSpatial&
    operator=(const GeoSpatial& other) {
        if (this != &other && other.IsValid()) {
            this->geometry_ = other.geometry_->clone();
            this->to_wkb_internal();
            this->to_wkt_internal();
        }
        return *this;
    }

    GeoSpatial&
    operator=(GeoSpatial&& other) noexcept {
        if (this != &other) {
            wkb_data_ro_ = std::move(other.wkb_data_ro_);
            wkb_data_wr_ = std::move(other.wkb_data_wr_);
            wkt_data_ = std::move(other.wkt_data_);
            geometry_ = std::move(other.geometry_);
        }
        return *this;
    }

    operator std::string() const {
        //tmp string created by copy ctr
        return std::string(wkt_data_);
    }

    ~GeoSpatial() {
        if (geometry_) {
            OGRGeometryFactory::destroyGeometry(geometry_);
        }
        if (wkb_data_wr_) {  // the data
            delete[] wkb_data_wr_;
        }
        wkt_data_.clear();
    }

    bool
    IsValid() const {
        return geometry_ != nullptr;
    }

    OGRGeometry*
    GetGeometry() const {
        return geometry_;
    }

    //only expose read only ptr to external env
    const unsigned char*
    wkb_data() const {
        return wkb_data_ro_;
    }

    size_t
    wkb_size() const {
        return size_;
    }

    std::string_view
    data() const {
        return wkt_data_;
    }

    const char*
    c_str() const {
        return wkt_data_.c_str();
    }

    size_t
    size() const {
        return wkt_data_.size();
    }

 private:
    inline void
    to_wkb_internal() {
        if (geometry_ && size_ == 0) {
            size_ = geometry_->WkbSize();
            wkb_data_wr_ = new unsigned char[size_];
            wkb_data_ro_ = wkb_data_wr_;
            geometry_->exportToWkb(wkbNDR, wkb_data_wr_);
        } else {
            LOG_WARN("[TH_DEBUG]change from geo to wkb failed!");
        }
    }

    inline void
    to_wkt_internal() {
        if (geometry_ && wkt_data_ == "") {
            wkt_data_ = geometry_->exportToWkt();
        } else {
            LOG_WARN("[TH_DEBUG]change from geo to wkt failed! ");
        }
    }

    //ready only ptr
    const unsigned char* wkb_data_ro_{nullptr};
    //read write ptr, use to save a OGRGeometry object when need to create new storage file
    unsigned char* wkb_data_wr_{nullptr};
    size_t size_ = 0;
    std::string wkt_data_ = "";
    OGRGeometry* geometry_{nullptr};
};

}  // namespace milvus