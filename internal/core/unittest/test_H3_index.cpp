#include <gtest/gtest.h>
#include <pthread.h>
#include <cstddef>
#include <functional>
#include <boost/filesystem.hpp>
#include <string>
#include <unordered_set>
#include <memory>
#include <vector>

#include "common/Tracer.h"
#include "h3/h3api.h"
#include "ogr_api.h"
#include "ogr_core.h"
#include "ogr_geometry.h"
#include "storage/Util.h"
#include "storage/InsertData.h"
#include "indexbuilder/IndexFactory.h"
#include "index/IndexFactory.h"
#include "index/H3Index.h"
#include "test_utils/indexbuilder_test_utils.h"
#include "index/Meta.h"
#include "pb/schema.pb.h"

using namespace milvus::index;
using namespace milvus::indexbuilder;
using namespace milvus;
using namespace milvus::index;

std::vector<std::string>
GenerateData(size_t n) {
    OGRPoint point(3.0, 4.0);
    OGRPoint point1(4.0, 4.0);
    OGRPoint point2(4.0, 5.0);
    OGRPoint point3(3.0, 5.0);
    OGRPoint point4(60.10, 40.10);
    OGRPoint point5(-40.00, -30.20);

    OGRLineString line;
    line.addPoint(&point);
    line.addPoint(&point1);
    line.addPoint(&point2);
    line.addPoint(&point3);
    OGRPolygon polygon;
    OGRLinearRing ring;
    ring.addPoint(&point);
    ring.addPoint(&point1);
    ring.addPoint(&point2);
    ring.addPoint(&point3);
    ring.addPoint(&point);
    polygon.addRing(&ring);
    size_t len1 = point.WkbSize(), len2 = line.WkbSize(),
           len3 = polygon.WkbSize(), len4 = point4.WkbSize(),
           len5 = point5.WkbSize();
    std::string str1, str2, str3, str4, str5;
    str1.resize(len1);
    str2.resize(len2);
    str3.resize(len3);
    str4.resize(len4);
    str5.resize(len5);
    point.exportToWkb(wkbNDR, reinterpret_cast<unsigned char*>(str1.data()));
    line.exportToWkb(wkbNDR, reinterpret_cast<unsigned char*>(str2.data()));
    polygon.exportToWkb(wkbNDR, reinterpret_cast<unsigned char*>(str3.data()));
    point4.exportToWkb(wkbNDR, reinterpret_cast<unsigned char*>(str4.data()));
    point5.exportToWkb(wkbNDR, reinterpret_cast<unsigned char*>(str5.data()));
    return {str1, str2, str3, str4, str5};
}

std::vector<std::string>
GenerateTestData(size_t n) {
    OGRPoint point(3.25, 3.75);
    OGRPoint point1(3.75, 3.75);
    OGRPoint point2(3.75, 4.25);
    OGRPoint point3(3.25, 4.25);
    OGRPolygon polygon;
    OGRLinearRing ring;
    ring.addPoint(&point);
    ring.addPoint(&point1);
    ring.addPoint(&point2);
    ring.addPoint(&point3);
    ring.addPoint(&point);
    polygon.addRing(&ring);
    size_t len1 = point.WkbSize(), len3 = polygon.WkbSize();
    std::string str1, str3;
    str1.resize(len1);
    str3.resize(len3);
    point.exportToWkb(wkbNDR, reinterpret_cast<unsigned char*>(str1.data()));
    polygon.exportToWkb(wkbNDR, reinterpret_cast<unsigned char*>(str3.data()));
    return {str1, str3};
}

class H3IndexTest : public testing::Test {
 protected:
    void
    Init(int64_t collection_id,
         int64_t partition_id,
         int64_t segment_id,
         int64_t field_id,
         int64_t index_build_id,
         int64_t index_version) {
        proto::schema::FieldSchema field_schema;
        field_schema.set_nullable(nullable_);
        field_schema.set_data_type(milvus::proto::schema::GeoSpatial);
        auto field_meta = storage::FieldDataMeta{
            collection_id, partition_id, segment_id, field_id, field_schema};
        auto index_meta = storage::IndexMeta{
            segment_id, field_id, index_build_id, index_version};

        std::vector<std::string> data_gen;
        data_gen = GenerateData(nb_);
        for (auto x : data_gen) {
            data_.push_back(x);
        }

        auto field_data = storage::CreateFieldData(type_, nullable_);
        if (nullable_) {
            valid_data_.reserve(nb_);
            uint8_t* ptr = new uint8_t[(nb_ + 7) / 8];
            for (int i = 0; i < nb_; i++) {
                int byteIndex = i / 8;
                int bitIndex = i % 8;
                if (i % 2 == 0) {
                    valid_data_.push_back(true);
                    ptr[byteIndex] |= (1 << bitIndex);
                } else {
                    valid_data_.push_back(false);
                    ptr[byteIndex] &= ~(1 << bitIndex);
                }
            }
            field_data->FillFieldData(data_.data(), ptr, data_.size());
            delete[] ptr;
        } else {
            field_data->FillFieldData(data_.data(), data_.size());
        }
        storage::InsertData insert_data(field_data);
        insert_data.SetFieldDataMeta(field_meta);
        insert_data.SetTimestamps(0, 100);

        auto serialized_bytes = insert_data.Serialize(storage::Remote);

        auto log_path = fmt::format("/{}/{}/{}/{}/{}/{}",
                                    "/tmp/test_H3/",
                                    collection_id,
                                    partition_id,
                                    segment_id,
                                    field_id,
                                    0);
        chunk_manager_->Write(
            log_path, serialized_bytes.data(), serialized_bytes.size());

        storage::FileManagerContext ctx(field_meta, index_meta, chunk_manager_);
        std::vector<std::string> index_files;

        Config config;
        config["index_type"] = milvus::index::H3_INDEX_TYPE;
        config["insert_files"] = std::vector<std::string>{log_path};

        auto build_index =
            indexbuilder::IndexFactory::GetInstance().CreateIndex(
                type_, config, ctx);
        build_index->Build();

        auto binary_set = build_index->Upload();
        for (const auto& [key, _] : binary_set.binary_map_) {
            index_files.push_back(key);
        }

        index::CreateIndexInfo index_info{};
        index_info.index_type = milvus::index::H3_INDEX_TYPE;
        index_info.field_type = type_;

        config["index_files"] = index_files;

        index_ =
            index::IndexFactory::GetInstance().CreateIndex(index_info, ctx);
        index_->Load(milvus::tracer::TraceContext{}, config);
    }

    virtual void
    SetParam() {
        nb_ = 5;
        nullable_ = false;
    }
    void
    SetUp() override {
        SetParam();
        type_ = DataType::GEOSPATIAL;
        int64_t collection_id = 1;
        int64_t partition_id = 2;
        int64_t segment_id = 3;
        int64_t field_id = 101;
        int64_t index_build_id = 1000;
        int64_t index_version = 10000;
        std::string root_path = "/tmp/test-H3-index";

        storage::StorageConfig storage_config;
        storage_config.storage_type = "local";
        storage_config.root_path = root_path;
        chunk_manager_ = storage::CreateChunkManager(storage_config);

        Init(collection_id,
             partition_id,
             segment_id,
             field_id,
             index_build_id,
             index_version);
    }
    virtual ~H3IndexTest() override {
        boost::filesystem::remove_all(chunk_manager_->GetRootPath());
    }

 public:
    void
    TestInFunc() {
        std::vector<std::string> test_data = GenerateTestData(2);
        auto index_ptr = dynamic_cast<GeoH3Index*>(index_.get());
        auto bitset = index_ptr->In(test_data.size(), test_data.data());
        for (size_t i = 0; i < bitset.size(); i++) {  //just test 5 elements
            if (nullable_ && !valid_data_[i]) {
                ASSERT_EQ(bitset[i], false);
            } else {
                if (i < 3) {
                    ASSERT_EQ(bitset[i], true);
                } else {
                    ASSERT_EQ(bitset[i], false);
                }
            }
        }
    }

    void
    TestNotInFunc() {
        std::vector<std::string> test_data = GenerateTestData(2);
        auto index_ptr = dynamic_cast<GeoH3Index*>(index_.get());
        auto bitset = index_ptr->In(test_data.size(), test_data.data());
        for (size_t i = 0; i < bitset.size(); i++) {
            if (nullable_ && !valid_data_[i]) {
                ASSERT_EQ(bitset[i], false);
            } else {
                if (i < 3) {
                    ASSERT_EQ(bitset[i], false);
                } else {
                    ASSERT_EQ(bitset[i], true);
                }
            }
        }
    }

    void
    TestIsNullFunc() {
        auto index_ptr = dynamic_cast<index::GeoH3Index*>(index_.get());
        auto bitset = index_ptr->IsNull();
        for (size_t i = 0; i < bitset.size(); i++) {
            if (nullable_ && !valid_data_[i]) {
                ASSERT_EQ(bitset[i], true);
            } else {
                ASSERT_EQ(bitset[i], false);
            }
        }
    }

    void
    TestIsNotNullFunc() {
        auto index_ptr = dynamic_cast<index::GeoH3Index*>(index_.get());
        auto bitset = index_ptr->IsNotNull();
        for (size_t i = 0; i < bitset.size(); i++) {
            if (nullable_ && !valid_data_[i]) {
                ASSERT_EQ(bitset[i], false);
            } else {
                ASSERT_EQ(bitset[i], true);
            }
        }
    }

    void
    TestExecGeoRelationFunc() {
        OGRPoint point(3.0, 4.0);
        std::string wkb;
        wkb.resize(point.WkbSize());
        point.exportToWkb(wkbNDR, reinterpret_cast<unsigned char*>(wkb.data()));
        auto index_ptr = dynamic_cast<index::GeoH3Index*>(index_.get());
        auto bitset = index_ptr->ExecGeoRelations(
            1,
            &wkb,
            proto::plan::GISFunctionFilterExpr_GISOp::
                GISFunctionFilterExpr_GISOp_Equals);
        for (size_t i = 0; i < bitset.size(); i++) {
            if (nullable_ && !valid_data_[i]) {
                ASSERT_EQ(bitset[i], false);
            } else {
                if (i == 0) {
                    ASSERT_EQ(bitset[i],
                              true);  //only the first wkb fits the condition
                } else {
                    ASSERT_EQ(bitset[i], false);
                }
            }
        }
    }

 public:
    IndexBasePtr index_;
    DataType type_;
    size_t nb_;
    std::vector<std::string> data_;
    std::shared_ptr<storage::ChunkManager> chunk_manager_;
    bool nullable_;
    FixedVector<bool> valid_data_;
};