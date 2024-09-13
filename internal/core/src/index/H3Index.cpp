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

#define INDEX_EXEC_GISOP(method)                                       \
    do {                                                               \
        for (int i = 0; i < n; ++i) {                                  \
            if (origin.method(                                         \
                    GeoSpatial(values[i].data(), values[i].size()))) { \
                res.set(idx.value());                                  \
                break;                                                 \
            }                                                          \
        }                                                              \
    } while (0)

GeoH3Index::GeoH3Index(const storage::FileManagerContext& file_manager_context,
                       int resolution)
    : is_built_(false),
      schema_(file_manager_context.fieldDataMeta.field_schema),
      resolution_(resolution) {
    AssertInfo(resolution_ < 16 && resolution_ >= 0,
               "Invalid max resolution to build H3 index");
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
    for (auto iter = index_data_.begin(); iter != index_data_.end(); ++iter) {
        res +=
            sizeof(H3Index) + sizeof(uint32_t) +
            iter->second->size() *
                sizeof(
                    uint32_t);  // H3Index ,num rows this index indexed, field offsets
    };
    for (auto& wkb_str : raw_data_) {
        res += sizeof(uint32_t) + wkb_str.size();  // wkb_str_size , wkb_data
    }
    return res;
}

void
GeoH3Index::DeserializeIndexData(const uint8_t* data_ptr, size_t index_length) {
    size_t pos = 0;
    raw_data_.resize(total_num_rows_);
    while (pos < index_length) {
        H3Index index = *reinterpret_cast<const H3Index*>(data_ptr + pos);
        pos += sizeof(H3Index);

        uint32_t vector_size =
            *reinterpret_cast<const uint32_t*>(data_ptr + pos);
        pos += sizeof(uint32_t);

        for (int i = 0; i < vector_size; ++i) {
            uint32_t offset =
                *reinterpret_cast<const uint32_t*>(data_ptr + pos);
            if (i == 0) {
                std::vector<uint32_t> indexed_offsets{offset};
                index_data_[index] =
                    std::make_unique<std::vector<uint32_t>>(indexed_offsets);
            } else {
                index_data_[index]->emplace_back(offset);
            }
            pos += sizeof(uint32_t);

            uint32_t wkb_byte_size =
                *reinterpret_cast<const uint32_t*>(data_ptr + pos);
            pos += sizeof(uint32_t);

            raw_data_[offset] = std::string(
                reinterpret_cast<const char*>(data_ptr + pos), wkb_byte_size);
            pos += wkb_byte_size;
        }
    }
}

int64_t
GeoH3Index::Cardinality() {
    return index_data_.size();
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
    for (auto iter = index_data_.begin(); iter != index_data_.end(); iter++) {
        std::memcpy(data_ptr, &iter->first, sizeof(H3Index));
        data_ptr += sizeof(H3Index);

        uint32_t vector_size(iter->second->size());
        std::memcpy(data_ptr, &vector_size, sizeof(uint32_t));
        data_ptr += sizeof(uint32_t);

        for (auto& offset : *(iter->second)) {
            std::memcpy(data_ptr, &offset, sizeof(uint32_t));
            data_ptr += sizeof(uint32_t);

            uint32_t wkb_byte_size = raw_data_[offset].size();
            std::memcpy(data_ptr, &wkb_byte_size, sizeof(uint32_t));
            data_ptr += sizeof(uint32_t);

            std::memcpy(data_ptr, raw_data_[offset].data(), wkb_byte_size);
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
    for (uint32_t offset = 0; offset < n; ++offset) {
        if (values[offset] == "") {  //null value
            null_offsets_.emplace_back(offset);
            continue;
        }
        OGRGeometry* geometry{nullptr};
        OGRGeometryFactory::createFromWkb(
            values[offset].data(), nullptr, &geometry, values[offset].size());
        H3Index cur_idx;
        getRepresentH3Index(cur_idx, geometry);
        if (index_data_.find(cur_idx) != index_data_.end()) {
            index_data_[cur_idx]->emplace_back(offset);
        } else {
            std::vector<uint32_t> res{offset};
            index_data_[cur_idx] = std::make_unique<std::vector<uint32_t>>(res);
        }
        raw_data_[offset] = values[offset];
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

//In just use a geometry's represent index to roughly filter out all shapes in the index
//that are either parent indices or child indices of the specified index.
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
        //obtain every parent resolution in range [0,rep_res],find the related parent H3Index
        for (int cur_res = rep_res; cur_res >= 0; --cur_res) {
            H3Index parent;
            H3Error err = cellToParent(rep_index, cur_res, &parent);
            AssertInfo(
                err = Success, "error when getting parent! the code {}", err);
            auto it = index_data_.find(parent);
            if (it != index_data_.end()) {
                for (auto& offset : *it->second) {
                    res.set(offset);
                }
            }
        }
        //obtain every child resolution in range [rep_res+1,resolution_],find the related children H3Indexes
        //this step may cause large search space,since the number of children of a small resolution may too large
        for (int cur_res = rep_res + 1; cur_res <= resolution_; ++cur_res) {
            int64_t child_num = 0;
            H3Error err = cellToChildrenSize(rep_index, cur_res, &child_num);
            AssertInfo(err == Success,
                       "error when getting children number! the code {}",
                       err);
            H3Index* children = new H3Index[child_num];
            err = cellToChildren(rep_res, cur_res, children);
            AssertInfo(err == Success,
                       "error when getting children! the code {}",
                       err);
            for (int i = 0; i < child_num; ++i) {
                auto it = index_data_.find(children[i]);
                if (it != index_data_.end()) {
                    for (auto& offset : *it->second) {
                        res.set(offset);
                    }
                }
            }
            delete[] children;
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
        //obtain every parent resolution in range [0,rep_res],find the related parent H3Index
        for (int cur_res = rep_res; cur_res >= 0; --cur_res) {
            H3Index parent;
            H3Error err = cellToParent(rep_index, cur_res, &parent);
            AssertInfo(
                err = Success, "error when getting parent! the code {}", err);
            auto it = index_data_.find(parent);
            if (it != index_data_.end()) {
                for (auto& offset : *it->second) {
                    res.set(offset, false);
                }
            }
        }
        //obtain every child resolution in range [rep_res+1,resolution_],find the related children H3Indexes
        for (int cur_res = rep_res + 1; cur_res <= resolution_; ++cur_res) {
            int64_t child_num = 0;
            H3Error err = cellToChildrenSize(rep_index, cur_res, &child_num);
            AssertInfo(err == Success,
                       "error when getting children number! the code {}",
                       err);
            H3Index* children = new H3Index[child_num];
            err = cellToChildren(rep_res, cur_res, children);
            AssertInfo(err == Success,
                       "error when getting children! the code {}",
                       err);
            for (int i = 0; i < child_num; ++i) {
                auto it = index_data_.find(children[i]);
                if (it != index_data_.end()) {
                    for (auto& offset : *it->second) {
                        res.set(offset, false);
                    }
                }
            }
            delete[] children;
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
    return raw_data_[offset];
}

const TargetBitmap
GeoH3Index::ExecGeoRelations(size_t n,
                             const std::string* values,
                             proto::plan::GISFunctionFilterExpr_GISOp op) {
    TargetBitmap Inmap = this->In(n, values);
    TargetBitmap res(this->Count(), false);
    for (std::optional<size_t> idx = Inmap.find_first(); idx.has_value();
         idx = Inmap.find_next(idx.value())) {
        GeoSpatial origin(raw_data_[idx.value()].data(),
                          raw_data_[idx.value()].size());
        switch (op) {
            case proto::plan::GISFunctionFilterExpr_GISOp::
                GISFunctionFilterExpr_GISOp_Equals: {
                INDEX_EXEC_GISOP(equals);
                break;
            }
            case proto::plan::GISFunctionFilterExpr_GISOp::
                GISFunctionFilterExpr_GISOp_Touches: {
                INDEX_EXEC_GISOP(touches);
                break;
            }
            case proto::plan::GISFunctionFilterExpr_GISOp::
                GISFunctionFilterExpr_GISOp_Overlaps: {
                INDEX_EXEC_GISOP(overlaps);
                break;
            }
            case proto::plan::GISFunctionFilterExpr_GISOp::
                GISFunctionFilterExpr_GISOp_Crosses: {
                INDEX_EXEC_GISOP(crosses);
                break;
            }
            case proto::plan::GISFunctionFilterExpr_GISOp::
                GISFunctionFilterExpr_GISOp_Contains: {
                INDEX_EXEC_GISOP(contains);
                break;
            }
            case proto::plan::GISFunctionFilterExpr_GISOp::
                GISFunctionFilterExpr_GISOp_Intersects: {
                INDEX_EXEC_GISOP(intersects);
                break;
            }
            case proto::plan::GISFunctionFilterExpr_GISOp::
                GISFunctionFilterExpr_GISOp_Within: {
                INDEX_EXEC_GISOP(within);
                break;
            }
            default: {
                PanicInfo(NotImplemented, "Invalid GIS Function op");
            }
        }
    }
    return res;
}

}  // namespace index
}  // namespace milvus
