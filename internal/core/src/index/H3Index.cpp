#include "index/H3Index.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "common/EasyAssert.h"
#include "common/Slice.h"
#include "common/Types.h"
#include "h3/h3api.h"
#include "log/Log.h"
#include "ogr_core.h"
#include "index/Utils.h"
#include "pb/schema.pb.h"
#include "index/Meta.h"

#include <gdal.h>
#include <ogr_geometry.h>

namespace milvus {
namespace index {

GeoH3Index::GeoH3Index(const storage::FileManagerContext& file_manager_context,
                       int resolution)
    : is_built_(false),
      schema_(file_manager_context.fieldDataMeta.field_schema),
      resolution_(resolution) {
    AssertInfo(resolution_ < 16 && resolution_ >= 0,
               "Invalid resolution to build H3 index");
    AssertInfo(schema_.data_type() == proto::schema::GeoSpatial,
               "Invalid type ,H3 only support built on GeoSpatial field");
    if (file_manager_context.Valid()) {
        file_manager_ =
            std::make_shared<storage::MemFileManagerImpl>(file_manager_context);
        AssertInfo(file_manager_ != nullptr, "create file manager failed!");
    }
}

size_t
GeoH3Index::GetIndexDataSize() {
    size_t res = 0;
    for (auto iter = data_.begin(); iter != data_.end(); ++iter) {
        res += sizeof(H3Index);
        for (auto pair : *(iter->second)) {
            res += (sizeof(pair.first) + pair.second.size());
        }
    }
    return res;
}
void
GeoH3Index::DeserializeIndexData(const uint8_t* data_ptr, size_t index_length) {
    size_t pos = 0;
    while (pos < index_length) {
        H3Index index = *reinterpret_cast<const H3Index*>(data_ptr + pos);
        pos += sizeof(H3Index);

        uint32_t vector_size =
            *reinterpret_cast<const uint32_t*>(data_ptr + pos);
        pos += sizeof(uint32_t);

        std::vector<std::pair<uint32_t, std::string>> res;
        for (int i = 0; i < vector_size; ++i) {
            uint32_t seg_offset =
                *reinterpret_cast<const uint32_t*>(data_ptr + pos);
            pos += sizeof(uint32_t);

            uint32_t wkb_byte_size =
                *reinterpret_cast<const uint32_t*>(data_ptr + pos);
            pos += sizeof(uint32_t);

            res.emplace_back(std::pair<uint32_t, std::string>(
                seg_offset,
                std::move(std::string(reinterpret_cast<const char*>(data_ptr),
                                      wkb_byte_size))));
            pos += wkb_byte_size;
        }
        data_[index] =
            std::make_unique<std::vector<std::pair<uint32_t, std::string>>>(
                res);
    }
}

int64_t
GeoH3Index::Cardinality() {
    return data_.size();
}

void
GeoH3Index::LoadWithoutAssemble(const BinarySet& binary_set,
                                const Config& config) {
    auto index_meta_buffer = binary_set.GetByName(H3_INDEX_NUM_ROWS);
    total_num_rows_ =
        *(reinterpret_cast<size_t*>(index_meta_buffer->data.get()));

    auto index_null_offsets = binary_set.GetByName(H3_INDEX_NULL_OFFSET);
    std::memcpy(null_offsets_.data(),
                index_null_offsets->data.get(),
                index_null_offsets.get()->size);

    auto index_data_buffer = binary_set.GetByName(H3_INDEX_DATA);
    DeserializeIndexData(index_data_buffer->data.get(),
                         index_data_buffer->size);
    LOG_INFO("load H3 index with cardinality = {}, num_rows = {}",
             Cardinality(),
             total_num_rows_);
    is_built_ = true;
}

void
GeoH3Index::SerializeIndexData(uint8_t* data_ptr) {
    for (auto iter = data_.begin(); iter != data_.end(); iter++) {
        std::memcpy(data_ptr, &iter->first, sizeof(H3Index));
        data_ptr += sizeof(H3Index);

        uint32_t vector_size(iter->second->size());
        std::memcpy(data_ptr, &vector_size, sizeof(uint32_t));
        data_ptr += sizeof(uint32_t);

        for (auto& pair : *(iter->second)) {
            uint32_t seg_offset(pair.first);
            std::memcpy(data_ptr, &seg_offset, sizeof(uint32_t));
            data_ptr += sizeof(uint32_t);

            uint32_t wkb_byte_size = pair.second.size();
            std::memcpy(data_ptr, &wkb_byte_size, sizeof(uint32_t));
            data_ptr += sizeof(uint32_t);

            std::memcpy(data_ptr, pair.second.data(), wkb_byte_size);
            data_ptr += wkb_byte_size;
        }
    }
}

//the index data designed to maintain raw geospatial data, organize as:
//index|raw data vector size|seg_offset1|raw data1 size|raw_data1|seg_offset2|raw data2 size|raw data2|...|...
BinarySet
GeoH3Index::Serialize(const Config& config) {
    AssertInfo(is_built_, "index has not been built yet");

    auto index_data_size = GetIndexDataSize();
    std::shared_ptr<uint8_t[]> index_data(new uint8_t[index_data_size]);
    uint8_t* data_ptr = index_data.get();
    SerializeIndexData(data_ptr);

    size_t byte_size = sizeof(int64_t) * null_offsets_.size();
    std::shared_ptr<uint8_t[]> index_null_offsets(new uint8_t[byte_size]);
    std::memcpy(index_null_offsets.get(), null_offsets_.data(), byte_size);

    std::shared_ptr<uint8_t[]> index_num_rows(new uint8_t[sizeof(size_t)]);
    memcpy(index_num_rows.get(), &total_num_rows_, sizeof(size_t));

    BinarySet res_set;
    res_set.Append(H3_INDEX_DATA, index_data, index_data_size);
    res_set.Append(H3_INDEX_NULL_OFFSET, index_null_offsets, byte_size);
    res_set.Append(H3_INDEX_NUM_ROWS, index_num_rows, sizeof(size_t));
    milvus::Disassemble(res_set);
    return res_set;
}

void
GeoH3Index::BuildWithFieldData(const std::vector<FieldDataPtr>& field_datas) {
    std::vector<std::string> all_datas;
    for (const auto& data : field_datas) {
        if (data->get_data_type() != DataType::GEOSPATIAL) {
            LOG_WARN("received data type is not geospatial");
            continue;
        }
        auto data_ptr = static_cast<std::string*>(data->Data());
        auto n = data->get_num_rows();
        for (int i = 0; i < n; ++i) {
            if (!data->is_valid(i)) {
                all_datas.emplace_back("");
            } else {
                all_datas.emplace_back(data_ptr[i]);
            }
        }
    }
    return Build(all_datas.size(), all_datas.data());
}

void
GeoH3Index::Load(const BinarySet& index_binary, const Config& config) {
    milvus::Assemble(const_cast<BinarySet&>(index_binary));
    LoadWithoutAssemble(index_binary, config);
}

void
GeoH3Index::Load(milvus::tracer::TraceContext ctx, const Config& config) {
    auto index_files =
        GetValueFromConfig<std::vector<std::string>>(config, "index_files");
    AssertInfo(index_files.has_value(),
               "index file paths is empty when load bitmap index");
    auto index_datas = file_manager_->LoadIndexToMemory(index_files.value());
    AssembleIndexDatas(index_datas);
    BinarySet binary_set;
    for (auto& [key, data] : index_datas) {
        auto size = data->DataSize();
        auto deleter = [&](uint8_t*) {};  // avoid repeated deconstruction
        auto buf = std::shared_ptr<uint8_t[]>(
            (uint8_t*)const_cast<void*>(data->Data()), deleter);
        binary_set.Append(key, buf, size);
    }
    LoadWithoutAssemble(binary_set, config);
}

void
GeoH3Index::getRepresentH3Index(H3Index& index, OGRGeometry* geometry) {
    AssertInfo(geometry != nullptr, "Geospatial Field data invalid!");
    switch (geometry->getGeometryType()) {
        case OGRwkbGeometryType::wkbPoint: {
            auto geoptr = static_cast<OGRPoint*>(geometry);
            LatLng pg{geoptr->getX(), geoptr->getY()};
            latLngToCell(&pg, resolution_, &index);
            AssertInfo(isValidCell(index), "error when convert latlng to cell");
            break;
        }
        case OGRwkbGeometryType::wkbLineString: {
            auto geoptr = static_cast<OGRLineString*>(geometry);
            OGRPoint* point{nullptr};
            LatLng pg{0, 0};
            int n = geoptr->getNumPoints();
            std::unordered_set<H3Index> out;
            for (int i = 0; i < n; ++i) {
                H3Index idx;
                geoptr->getPoint(i, point);
                pg = {point->getX(), point->getY()};
                latLngToCell(&pg, resolution_, &idx);
                AssertInfo(isValidCell(idx),
                           "error when convert latlng to cell");
                out.emplace(idx);
            }
            int res = resolution_;
            while (out.size() != 1) {
                --res;
                std::unordered_set<H3Index> new_out;
                for (H3Index val : out) {
                    H3Index parent;
                    cellToParent(val, res, &parent);
                    AssertInfo(isValidCell(parent),
                               "error when convert latlng to cell");
                    new_out.emplace(parent);
                }
                out.swap(new_out);
            }
            index = *out.begin();
            break;
        }
        case OGRwkbGeometryType::wkbPolygon: {
            auto geoptr = static_cast<OGRPolygon*>(geometry);
            GeoPolygon polygon;
            OGRLinearRing* exteriorRing = geoptr->getExteriorRing();
            polygon.geoloop.numVerts = exteriorRing->getNumPoints();
            polygon.geoloop.verts = new LatLng[polygon.geoloop.numVerts];
            for (int i = 0; i < polygon.geoloop.numVerts; ++i) {
                OGRPoint* point{nullptr};
                exteriorRing->getPoint(i, point);
                polygon.geoloop.verts[i] = LatLng{point->getX(), point->getY()};
            }
            polygon.numHoles = geoptr->getNumInteriorRings();
            if (polygon.numHoles) {
                polygon.holes = new GeoLoop[polygon.numHoles];
                for (int i = 0; i < polygon.numHoles; ++i) {
                    OGRLinearRing* interiorRing = geoptr->getInteriorRing(i);
                    polygon.holes[i].numVerts = interiorRing->getNumPoints();
                    polygon.holes[i].verts =
                        new LatLng[polygon.holes[i].numVerts];
                    for (int j = 0; j < polygon.holes[i].numVerts; ++j) {
                        OGRPoint* point{nullptr};
                        interiorRing->getPoint(j, point);
                        polygon.holes[i].verts[j] =
                            LatLng{point->getX(), point->getY()};
                    }
                }
            }
            int64_t num_H3indexs{0};
            maxPolygonToCellsSize(&polygon, resolution_, 0, &num_H3indexs);
            std::vector<H3Index> out;
            out.resize(num_H3indexs);
            H3Error err = polygonToCells(&polygon, resolution_, 0, out.data());
            AssertInfo(err = Success, "obtain the polygon covered area ");
            std::unordered_set<H3Index> out_set;
            for (H3Index idx : out) {
                out_set.emplace(idx);
            }
            int res = resolution_;
            while (out_set.size() != 1) {
                --res;
                std::unordered_set<H3Index> new_set;
                for (H3Index val : out_set) {
                    H3Index parent;
                    cellToParent(val, res, &parent);
                    AssertInfo(isValidCell(parent),
                               "error when convert latlng to cell");
                    new_set.emplace(parent);
                }
                out_set.swap(new_set);
            }
            index = *out_set.begin();
            for (int i = 0; i < polygon.numHoles; ++i) {
                delete[] polygon.holes[i].verts;
            }
            if (polygon.holes != nullptr) {
                delete[] polygon.holes;
            }
            delete[] polygon.geoloop.verts;
            break;
        }
        default: {
            PanicInfo(NotImplemented,
                      "indexing on geospatial only support on point, "
                      "linestring, polygon now!");
        }
    }
}

//temp:build from wkb's std::string
void
GeoH3Index::Build(size_t n, const std::string* values) {
    for (size_t seg_offset = 0; seg_offset < n; ++seg_offset) {
        if (values[seg_offset] == "") {  //null value
            null_offsets_.emplace_back(seg_offset);
            continue;
        }
        OGRGeometry* geometry{nullptr};
        OGRGeometryFactory::createFromWkb(values[seg_offset].data(),
                                          nullptr,
                                          &geometry,
                                          values[seg_offset].size());
        H3Index cur_idx;
        getRepresentH3Index(cur_idx, geometry);
        if (data_.find(cur_idx) != data_.end()) {
            data_[cur_idx]->emplace_back(std::pair<uint32_t, std::string>(
                seg_offset, values[seg_offset]));
        } else {
            std::vector<std::pair<uint32_t, std::string>> res;
            res.emplace_back(std::pair<uint32_t, std::string>(
                seg_offset, values[seg_offset]));
            data_[cur_idx] =
                std::make_unique<std::vector<std::pair<uint32_t, std::string>>>(
                    res);
        }
    }
    is_built_ = true;
    total_num_rows_ = n;
}

void
GeoH3Index::Build(const Config& config) {
    if (is_built_) {
        return;
    }
    auto insert_files =
        GetValueFromConfig<std::vector<std::string>>(config, "insert_files");
    AssertInfo(insert_files.has_value(),
               "insert file paths is empty when build index");

    auto field_datas =
        file_manager_->CacheRawDataToMemory(insert_files.value());
    BuildWithFieldData(field_datas);
}

//In just renturn ture if one wkb data has the same H3Index with the given wkb data in their min represent resolution
const TargetBitmap
GeoH3Index::In(size_t n, const std::string* values) {
    AssertInfo(is_built_, "index has not been built");
    TargetBitmap res(total_num_rows_, false);
    for (int i = 0; i < n; ++i) {
        OGRGeometry* geometry{nullptr};
        OGRGeometryFactory::createFromWkb(
            values[i].data(), nullptr, &geometry, values[i].size());
        H3Index rep_index;
        getRepresentH3Index(rep_index, geometry);
        int rep_res = getResolution(rep_index);
        //obtain every resolution in range [0,rep_res],find the related
        for (int cur_res = rep_res; cur_res >= 0; --cur_res) {
            H3Index parent;
            H3Error err = cellToParent(rep_index, cur_res, &parent);
            AssertInfo(
                err = Success, "error when getting parent! the code {}", err);
            auto it = data_.find(parent);
            if (it != data_.end()) {
                for (auto& pair : *it->second) {
                    res.set(pair.first);
                }
            }
        }
    }
    return res;
}

const TargetBitmap
GeoH3Index::NotIn(size_t n, const std::string* values) {
    AssertInfo(is_built_, "index has not been built");
    TargetBitmap res(total_num_rows_, true);
    for (int i = 0; i < n; ++i) {
        OGRGeometry* geometry{nullptr};
        OGRGeometryFactory::createFromWkb(
            values[i].data(), nullptr, &geometry, values[i].size());
        H3Index rep_index;
        getRepresentH3Index(rep_index, geometry);
        int rep_res = getResolution(rep_index);
        for (int cur_res = rep_res; cur_res >= 0; --cur_res) {
            H3Index parent;
            H3Error err = cellToParent(rep_index, cur_res, &parent);
            AssertInfo(
                err = Success, "error when getting parent! the code {}", err);
            auto it = data_.find(parent);
            if (it != data_.end()) {
                for (auto& pair : *it->second) {
                    res.set(pair.first, false);
                }
            }
        }
    }
    return res;
}

const TargetBitmap
GeoH3Index::IsNull() {
    AssertInfo(is_built_, "index has not been built");
    TargetBitmap res(total_num_rows_, false);
    for (auto offset : null_offsets_) {
        res.set(offset);
    }
    return res;
}

const TargetBitmap
GeoH3Index::IsNotNull() {
    AssertInfo(is_built_, "index has not been built");
    TargetBitmap res(total_num_rows_, true);
    for (auto offset : null_offsets_) {
        res.set(offset, false);
    }
    return res;
}

BinarySet
GeoH3Index::Upload(const Config& config) {
    auto binary_set = Serialize(config);
    file_manager_->AddFile(binary_set);

    auto remote_paths_to_size = file_manager_->GetRemotePathsToFileSize();
    BinarySet ret;
    for (auto& file : remote_paths_to_size) {
        ret.Append(file.first, nullptr, file.second);
    }

    return ret;
}

std::string
GeoH3Index::Reverse_Lookup(size_t offset) const {
    AssertInfo(is_built_, "index has not been built");
    AssertInfo(offset < total_num_rows_, "out of range of total coun");
    for (auto it = data_.begin(); it != data_.end(); ++it) {
        for (auto& pair : *it->second) {
            if (offset == pair.first) {
                return pair.second;
            }
        }
    }
}

}  // namespace index
}  // namespace milvus
