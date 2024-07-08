#include <gdal.h>
#include <ogr_geometry.h>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include "log/Log.h"

class GeoSpatial {
 public:
    GeoSpatial() = default;

    // all ctr assume that wkb data and wkt string is valid
    explicit GeoSpatial(const std::string& wkt) : wkt_data_(wkt) {
        OGRGeometryFactory::createFromWkt(wkt.c_str(), nullptr, &geometry_);
        if (!geometry_) {
            LOG_WARN("[TH_DEBUG]ctr geospatial failed from wkt");
        }
    }

    explicit GeoSpatial(const void* wkb, size_t size)
        : wkb_data_(static_cast<const unsigned char*>(wkb)), size_(size) {
        OGRGeometryFactory::createFromWkb(wkb, nullptr, &geometry_, size);
        if (!geometry_) {
            LOG_WARN("[TH_DEBUG]ctr geospatial failed from wkb");
        }
    }

    GeoSpatial(const GeoSpatial& other) {
        if (other.IsValid()) {
            // wkb_data_ = new unsigned char[other.size_];
            // std::copy(
            //     other.wkb_data_, other.wkb_data_ + other.size_, wkb_data_);
            // wkt_data_ = other.wkt_data_;
            this->geometry_ = other.geometry_->clone();
        }
    }

    GeoSpatial(GeoSpatial&& other) noexcept
        : wkb_data_(std::move(other.wkb_data_)),
          wkt_data_(std::move(other.wkt_data_)),
          geometry_(std::move(other.geometry_)) {
    }

    GeoSpatial&
    operator=(const GeoSpatial& other) {
        if (this != &other && other.IsValid()) {
            if (other.geometry_) {
                this->geometry_ = other.geometry_->clone();
            }
        }
        return *this;
    }

    GeoSpatial&
    operator=(GeoSpatial&& other) noexcept {
        if (this != &other) {
            wkb_data_ = std::move(other.wkb_data_);
            wkt_data_ = std::move(other.wkt_data_);
            geometry_ = std::move(other.geometry_);
        }
        return *this;
    }

    ~GeoSpatial() {
        if (geometry_) {
            OGRGeometryFactory::destroyGeometry(geometry_);
        }
        if (wkb_data_) {
            delete[] wkb_data_;
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

    std::string
    to_wkt() const {
        char* wkt = nullptr;
        if (geometry_ && geometry_->exportToWkt(&wkt) == OGRERR_NONE) {
            std::string result(wkt);
            CPLFree(wkt);
            return result;
        }
        return "";
    }

    void
    to_wkt_internal() {
        if (geometry_ && wkt_data_ == "") {
            wkt_data_ = geometry_->exportToWkt();
        }
    }

    std::vector<unsigned char>
    to_wkb() const {
        std::vector<unsigned char> wkb;
        if (geometry_) {
            size_t size = geometry_->WkbSize();
            wkb.resize(size);
            if (geometry_->exportToWkb(wkbNDR, wkb.data()) != OGRERR_NONE) {
                wkb.clear();
            }
        }
        return wkb;
    }

    // void
    // to_wkb_internal() {
    //     if (geometry_ && size_ == 0) {
    //         std::vector<unsigned char> wkb_data = geometry_->exportToWkb();
    //     }
    // }

    const unsigned char*
    wkb_data() const {
        return wkb_data_;
    }

    size_t
    wkb_size() const {
        return size_;
    }

    std::string
    wkt_data() const {
        return wkt_data_;
    }

    const char*
    wkt_data_c_str() const {
        return wkt_data_.c_str();
    }

    size_t
    wkt_size() const {
        return wkt_data_.size();
    }

 private:
    const unsigned char* wkb_data_{nullptr};
    size_t size_ = 0;
    std::string wkt_data_ = "";
    OGRGeometry* geometry_{nullptr};
};