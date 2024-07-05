package indexparamcheck

import (
	"fmt"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type H3Checker struct {
	scalarIndexChecker
}

func (c *H3Checker) CheckTrain(params map[string]string) error {
	return c.scalarIndexChecker.CheckTrain(params)
}

func (c *H3Checker) CheckValidDataType(field *schemapb.FieldSchema) error {
	if !typeutil.IsGeospatialType(field.GetDataType()) {
		return fmt.Errorf("H3 are only supported on geospatial field")
	}
	return nil
}

func newH3Checker() *H3Checker {
	return &H3Checker{}
}
