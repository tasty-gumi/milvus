#include "common/EasyAssert.h"
#include "index/ScalarIndex.h"
#include "pb/plan.pb.h"
#include "storage/MemFileManagerImpl.h"
#include <h3/h3api.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace milvus {
namespace index {

//Geospatial data storage as wkb bytes ,which can be storage as std::string in memory, the template parameter
//temporarily pass as std::string in ScalarIndex
class GeoH3Index : public ScalarIndex<std::string> {
 public:
    explicit GeoH3Index(
        const storage::FileManagerContext& file_manager_context =
            storage::FileManagerContext(),
        int resolution = 9);

    ~GeoH3Index() override = default;

    BinarySet
    Serialize(const Config& config) override;

    void
    Load(const BinarySet& index_binary, const Config& config = {}) override;

    void
    Load(milvus::tracer::TraceContext ctx, const Config& config = {}) override;

    ScalarIndexType
    GetIndexType() const override {
        return ScalarIndexType::H3;
    }

    void
    Build(size_t n, const std::string* values) override;

    void
    Build(const Config& config = {}) override;

    void
    BuildWithFieldData(const std::vector<FieldDataPtr>& field_datas) override;

    void
    LoadWithoutAssemble(const BinarySet& binary_set,
                        const Config& config) override;

    int64_t
    Cardinality();

    const TargetBitmap
    In(size_t n, const std::string* values) override;

    const TargetBitmap
    NotIn(size_t n, const std::string* values) override;

    const TargetBitmap
    IsNull() override;

    const TargetBitmap
    IsNotNull() override;

    const TargetBitmap
    Range(std::string value, OpType op) override {
        PanicInfo(NotImplemented, "Geospatial data do not support Range query");
    }

    const TargetBitmap
    Range(std::string lower_bound_value,
          bool lb_inclusive,
          std::string upper_bound_value,
          bool ub_inclusive) override {
        PanicInfo(NotImplemented, "Geospatial data do not support Range query");
    }

    std::string
    Reverse_Lookup(size_t offset) const override;

    int64_t
    Count() override {
        return total_num_rows_;
    }

    int64_t
    Size() override {
        return Count();
    }

    BinarySet
    Upload(const Config& config = {}) override;

    const bool
    HasRawData() const override {
        return true;
    }

    // use H3 index to accelerate spatial relation fliter
    const TargetBitmap
    ExecGeoRelations(size_t n,
                     const std::string* values,
                     proto::plan::GISFunctionFilterExpr_GISOp op);

 private:
    size_t
    GetIndexDataSize();

    void
    SerializeIndexData(uint8_t* data_ptr);

    void
    DeserializeIndexData(const uint8_t* data_ptr, size_t index_length);

    void
    getRepresentH3Index(H3Index& index, OGRGeometry* geometry);

 public:
    bool is_built_{false};
    proto::schema::FieldSchema schema_;
    //store hexagonal grid index provide by H3 system which can contain all points of a geometry and has the biggest resolution
    //map it to the set of field offsets
    std::unordered_map<H3Index, std::unique_ptr<std::vector<uint32_t>>>
        index_data_;
    std::vector<std::string> raw_data_;
    std::shared_ptr<storage::MemFileManagerImpl> file_manager_;
    int resolution_{0};         //max resolution of all index
    size_t total_num_rows_{0};  // the number of rows has been indexed
    std::vector<size_t> null_offsets_{};
};

}  // namespace index
}  // namespace milvus