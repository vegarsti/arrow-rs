// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::arrow::array_reader::ArrayReader;
use crate::errors::{ParquetError, Result};
use arrow_array::{Array, ArrayRef, Int16Array, Int32Array, Int64Array, RunArray, UInt32Array};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType as ArrowType;
use std::any::Any;
use std::sync::Arc;

/// An [`ArrayReader`] that reconstructs a run-end encoded array from a reader
/// for the decoded values.
pub struct RunEndEncodedArrayReader {
    data_type: ArrowType,
    inner: Box<dyn ArrayReader>,
}

impl RunEndEncodedArrayReader {
    pub fn new(data_type: ArrowType, inner: Box<dyn ArrayReader>) -> Self {
        Self { data_type, inner }
    }
}

impl ArrayReader for RunEndEncodedArrayReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        self.inner.read_records(batch_size)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        let values = self.inner.consume_batch()?;
        encode_run_end_encoded(values, &self.data_type)
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        self.inner.skip_records(num_records)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.inner.get_def_levels()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.inner.get_rep_levels()
    }
}

fn encode_run_end_encoded(values: ArrayRef, data_type: &ArrowType) -> Result<ArrayRef> {
    let ArrowType::RunEndEncoded(run_end_field, value_field) = data_type else {
        return Ok(values);
    };

    debug_assert_eq!(values.data_type(), value_field.data_type());

    let len = values.len();
    let mut run_starts = vec![];
    let mut run_ends = vec![];

    if len != 0 {
        run_starts.push(0_u32);

        for i in 1..len {
            if !same_value(values.as_ref(), i - 1, i) {
                run_ends.push(i);
                run_starts.push(i as u32);
            }
        }

        run_ends.push(len);
    }

    let indices = UInt32Array::from(run_starts);
    let run_values = arrow_select::take::take(values.as_ref(), &indices, None)?;

    match run_end_field.data_type() {
        ArrowType::Int16 => {
            let run_ends = run_ends
                .into_iter()
                .map(i16::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            make_run_array::<arrow_array::types::Int16Type>(
                Int16Array::from(run_ends),
                run_values,
                data_type.clone(),
                len,
            )
        }
        ArrowType::Int32 => {
            let run_ends = run_ends
                .into_iter()
                .map(i32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            make_run_array::<arrow_array::types::Int32Type>(
                Int32Array::from(run_ends),
                run_values,
                data_type.clone(),
                len,
            )
        }
        ArrowType::Int64 => {
            let run_ends = run_ends.into_iter().map(|x| x as i64).collect::<Vec<_>>();
            make_run_array::<arrow_array::types::Int64Type>(
                Int64Array::from(run_ends),
                run_values,
                data_type.clone(),
                len,
            )
        }
        other => Err(general_err!(
            "unsupported run-end type for RunEndEncoded: {}",
            other
        )),
    }
}

fn make_run_array<R>(
    run_ends: arrow_array::PrimitiveArray<R>,
    values: ArrayRef,
    data_type: ArrowType,
    len: usize,
) -> Result<ArrayRef>
where
    R: arrow_array::types::RunEndIndexType,
{
    let data = ArrayDataBuilder::new(data_type)
        .len(len)
        .add_child_data(run_ends.to_data())
        .add_child_data(values.to_data())
        .build()?;

    Ok(Arc::new(RunArray::<R>::from(data)))
}

fn same_value(values: &dyn Array, a: usize, b: usize) -> bool {
    let a_null = values.is_null(a);
    let b_null = values.is_null(b);

    if a_null || b_null {
        return a_null == b_null;
    }

    values.slice(a, 1).to_data() == values.slice(b, 1).to_data()
}
