package botkop

import java.math.BigInteger

import com.typesafe.scalalogging.LazyLogging
import torch.cpu._

import scala.language.implicitConversions

package object numsca extends LazyLogging {

  type Shape = List[Int]

  /* === numsca range == */
  case class NumscaRange(f: Int, t: Option[Int])

  def :>(end: Int) = NumscaRange(0, Some(end))
  def :> = NumscaRange(0, None)

  implicit class NumscaInt(i: Int) {
    def :>(end: Int) = NumscaRange(i, Some(end))
    def :> = NumscaRange(i, None)
  }

  implicit def intToNumscaRange(i: Int): NumscaRange =
    NumscaRange(i, Some(i + 1))

  case class NumscaRangeSeq(rs: Seq[NumscaRange])

  /* === random ================================================================================= */
  val rng: SWIGTYPE_p_THGenerator = TH.THGenerator_new()

  //  def initialSeed: Long = TH.THRandom_initialSeed(rng).longValue()
  //  def currentSeed: Long = TH.THRandom_seed(rng).longValue()
  def setSeed(theSeed: Long): Unit =
    TH.THRandom_manualSeed(rng, BigInteger.valueOf(theSeed))

  /* === tensor creation ================================================================================= */
  def copy(a: Tensor): Tensor = {
    val t = TH.THFloatTensor_newClone(a.payload)
    new Tensor(t)
  }

  def empty = new Tensor(TH.THFloatTensor_new())
  def array(data: Float*): Tensor = create(data: _*)
  def tensor(data: Float*): Tensor = create(data: _*)

  def create(data: Array[Float], shape: List[Int]): Tensor = {
    require(data.length == shape.product)
    val size = data.length
    val a = floatArray(data)
    val storage: THFloatStorage = TH.THFloatStorage_newWithData(a, size)
    val t =
      TH.THFloatTensor_newWithStorage(storage, 0, longStorage(shape), null)
    new Tensor(t)
  }

  def create(data: Array[Float]): Tensor = create(data, List(data.length))

  def create(data: Float*): Tensor = create(data.toArray)

  def zeros(shape: List[Int]): Tensor = {
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_zeros(t, ls)
    new Tensor(t)
  }

  def zeros(shape: Int*): Tensor = zeros(shape.toList)
  def zerosLike(other: Tensor): Tensor = zeros(other.shape)

  def ones(shape: List[Int]): Tensor = {
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_ones(t, ls)
    new Tensor(t)
  }

  def ones(shape: Int*): Tensor = ones(shape.toList)

  def fill(f: Float, shape: List[Int]): Tensor = {
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_fill(t, f)
    new Tensor(t)
  }

  def arange(min: Double = 0, max: Double, step: Double = 1): Tensor = {
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_arange(t, min, max, step)
    new Tensor(t)
  }

  def randn(shape: List[Int]): Tensor = {
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_randn(t, rng, ls)
    new Tensor(t)
  }

  def randn(shape: Int*): Tensor = randn(shape.toList)

  def randint(low: Double = 0.0, high: Double, shape: List[Int]): Tensor = {
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_uniform(t, rng, low, high)
    TH.THFloatTensor_floor(t, t)
    new Tensor(t)
  }

  def uniform(low: Double = 0.0, high: Double, shape: List[Int]): Tensor = {
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_uniform(t, rng, low, high)
    new Tensor(t)
  }

  def linspace(start: Float, end: Float, steps: Long): Tensor = {
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_linspace(t, start, end, steps)
    new Tensor(t)
  }

  // utility functions -----------------------------

  def longStorage(shape: Seq[Int]): THLongStorage = {
    val size = shape.length
    val data = new CInt64Array(size)
    var i = 0
    while (i < shape.length) {
      data.setitem(i, shape(i))
      i = i + 1
    }
    TH.THLongStorage_newWithData(data.cast(), size)
  }

  def floatArray(data: Array[Float]): SWIGTYPE_p_float = {
    val size = data.length
    val a = new CFloatArray(size)
    var i = 0
    while (i < size) {
      a.setitem(i, data(i))
      i = i + 1
    }
    a.cast()
  }

  def reshape(a: Tensor, newShape: List[Int]): Tensor = {
    val t = TH.THFloatTensor_newWithStorage(a.payload.getStorage,
      a.payload.getStorageOffset,
      longStorage(newShape),
      null)
    //     this creates a new storage tensor
    //    val t = TH.THFloatTensor_new()
    //    TH.THFloatTensor_reshape(t, a.payload, longStorage(newShape))
    new Tensor(t)
  }

  def reshape(a: Tensor, newShape: Int*): Tensor = reshape(a, newShape.toList)

  def setValue(a: Tensor, value: Float, index: List[Int]): Unit = {
    a.dim match {
      case 1 =>
        TH.THFloatTensor_set1d(a.payload, index.head, value)
      case 2 =>
        TH.THFloatTensor_set2d(a.payload, index.head, index(1), value)
      case 3 =>
        TH.THFloatTensor_set3d(a.payload, index.head, index(1), index(2), value)
      case 4 =>
        TH.THFloatTensor_set4d(a.payload,
          index.head,
          index(1),
          index(2),
          index(3),
          value)
    }
  }

  def getValue(a: Tensor, index: List[Int]): Float = {
    a.dim match {
      case 1 =>
        TH.THFloatTensor_get1d(a.payload, index.head)
      case 2 =>
        TH.THFloatTensor_get2d(a.payload, index.head, index(1))
      case 3 =>
        TH.THFloatTensor_get3d(a.payload, index.head, index(1), index(2))
      case 4 =>
        TH.THFloatTensor_get4d(a.payload,
          index.head,
          index(1),
          index(2),
          index(3))
    }
  }

  def select(a: Tensor, dimension: Int, sliceIndex: Int): Tensor = {
    val t = TH.THFloatTensor_newSelect(a.payload, dimension, sliceIndex)
    new Tensor(t)
  }

  def narrow(a: Tensor, where: List[Int]): Tensor = {
    val r = where.zipWithIndex.foldLeft(a.payload) {
      case (t, (i, d)) =>
        TH.THFloatTensor_newNarrow(t, d, i, 1)
    }

    val s = TH.THFloatTensor_new()
    TH.THFloatTensor_squeeze(s, r)
    new Tensor(s)
  }

  def narrow(a: Tensor, ranges: NumscaRangeSeq): Tensor = {
    val r = ranges.rs.zipWithIndex.foldLeft(a.payload) {
      case (t, (i, d)) =>
        val to = i.t match {
          case None =>
            TH.THFloatTensor_size(t, d).toInt
          case Some(n) if n < 0 =>
            TH.THFloatTensor_size(t, d).toInt + n
          case o =>
            o.get
        }

        val size = to - i.f
        TH.THFloatTensor_newNarrow(t, d, i.f, size)
    }

    val s = TH.THFloatTensor_new()
    TH.THFloatTensor_squeeze(s, r)
    new Tensor(s)
  }

  def assign(a: Tensor, f: Float): Tensor = {
    TH.THFloatTensor_fill(a.payload, f)
    a
  }

  def assign(a: Tensor, src: Tensor): Unit = {
    val t =
      if (a.size == src.size) {
        logger.debug("not broadcasting")
        src.payload
      } else { // broadcast
        logger.debug("broadcasting")
        TH.THFloatTensor_newExpand(src.payload, longStorage(a.shape))
      }
    TH.THFloatTensor_copy(a.payload, t)
  }

  def squeeze(a: Tensor): Tensor = {
    val s = TH.THFloatTensor_new()
    TH.THFloatTensor_squeeze(s, a.payload)
    new Tensor(s)
  }

  def narrow(a: Tensor,
             dimension: Int,
             firstIndex: Int,
             size: Int): Tensor = {
    val t = TH.THFloatTensor_newNarrow(a.payload, dimension, firstIndex, size)
    new Tensor(t)
  }

  def linear(x: Tensor, y: Tensor, b: Tensor): Tensor = {
    val t = TH.THFloatTensor_new()
    TH.THFloatTensor_addmm(t, 1.0f, b.payload, 1.0f, x.payload, y.payload)
    new Tensor(t)
  }

  def mul(t: Tensor, f: Float): Tensor = {
    val r = TH.THFloatTensor_new()
    TH.THFloatTensor_mul(r, t.payload, f)
    new Tensor(r)
  }

  def addi(t: Tensor, f: Float): Unit = {
    TH.THFloatTensor_add(t.payload, t.payload, f)
  }

  def addi(a: Tensor, b: Tensor): Unit = {
    TH.THFloatTensor_cadd(a.payload, a.payload, 1, b.payload)
  }

  def subi(t: Tensor, f: Float): Unit = {
    TH.THFloatTensor_sub(t.payload, t.payload, f)
  }

  def equal(a: Tensor, b: Tensor): Boolean =
    TH.THFloatTensor_equal(a.payload, b.payload) == 1

  /*
  one ops
   */
  def oneOp(f: (THFloatTensor, THFloatTensor) => Unit, a: Tensor): Tensor = {
    val r = TH.THFloatTensor_new()
    f(r, a.payload)
    new Tensor(r)
  }

  def neg(a: Tensor): Tensor = oneOp(TH.THFloatTensor_neg, a)
  def sqrt(a: Tensor): Tensor = oneOp(TH.THFloatTensor_sqrt, a)
  def square(a: Tensor): Tensor =
    oneOp((r, t) => TH.THFloatTensor_pow(r, t, 2), a)

  /*
  bin ops
   */
  def binOp(f: (THFloatTensor, THFloatTensor, THFloatTensor) => Unit,
            a: Tensor,
            b: Tensor): Tensor = {
    val r = TH.THFloatTensor_new()
    val Seq(ta, tb) = expandNd(Seq(a, b))
    f(r, ta.payload, tb.payload)
    new Tensor(r)
  }

  def mul(t1: Tensor, t2: Tensor): Tensor =
    binOp(TH.THFloatTensor_cmul, t1, t2)

  def sub(a: Tensor, b: Tensor): Tensor = {
    binOp((r, t, u) => TH.THFloatTensor_csub(r, t, 1, u), a, b)
  }

  def add(a: Tensor, b: Tensor): Tensor = {
    binOp((r, t, u) => TH.THFloatTensor_cadd(r, t, 1, u), a, b)
  }

  //==================================================
  def sum(a: Tensor, axis: Int, keepDim: Boolean = true): Tensor = {
    val r = TH.THFloatTensor_new()
    val nAxis = if (axis < 0) a.dim + axis else axis
    TH.THFloatTensor_sum(r, a.payload, nAxis, if (keepDim) 1 else 0)
    new Tensor(r)
  }

  def sum(a: Tensor): Double = TH.THFloatTensor_sumall(a.payload)

  def argmin(a: Tensor, axis: Int, keepDim: Boolean = true): Tensor = {
    val values = TH.THFloatTensor_new()
    val indices = TH.THLongTensor_new()
    TH.THFloatTensor_min(values,
      indices,
      a.payload,
      axis,
      if (keepDim) 1 else 0)

    val indexSize =
      TH.THLongStorage_newWithData(indices.getSize, indices.getNDimension)

    // transform to floats
    val t = TH.THFloatTensor_newWithSize(indexSize, null)
    TH.THFloatTensor_copyLong(t, indices)

    TH.THFloatTensor_free(values)
    TH.THLongTensor_free(indices)
    new Tensor(t)
  }

  def expandNd(as: Seq[Tensor]): Seq[Tensor] =
    if (as.tail.forall(_.shape == as.head.shape)) {
      as
    } else {
      val original = TH.new_CFloatTensorArray(as.length)
      as.indices.foreach { i =>
        TH.CFloatTensorArray_setitem(original, i, as(i).payload)
      }

      val results = TH.new_CFloatTensorArray(as.length)
      as.indices.foreach { i =>
        val t = TH.THFloatTensor_new()
        TH.CFloatTensorArray_setitem(results, i, t)
      }

      TH.THFloatTensor_expandNd(results, original, as.length)
      val resized = as.indices.foldLeft(Seq.empty[Tensor]) {
        case (rs, i) =>
          rs :+ new Tensor(TH.CFloatTensorArray_getitem(results, i))
      }

      TH.delete_CFloatTensorArray(original)
      TH.delete_CFloatTensorArray(results)

      resized
    }

  def indexSelect(a: Tensor, dim: Int, ix: Tensor): Tensor = {
    // public static void THFloatTensor_indexSelect(THFloatTensor tensor, THFloatTensor src, int dim, THLongTensor index)

    // transform to long
    val index =
      TH.THLongTensor_newWithSize(longStorage(ix.shape), longStorage(ix.stride))
    TH.THLongTensor_copyFloat(index, ix.payload)

    val r = TH.THFloatTensor_new()
    TH.THFloatTensor_indexSelect(r, a.payload, dim, index)
    val result = new Tensor(r)

    TH.THLongTensor_free(index)
    result
  }

  def indexSelect(a: Tensor, ixs: Seq[Tensor]): Tensor = ixs.indices.foldLeft(a) {
    case (acc, i) =>
      val r = indexSelect(acc, 0, ixs(i))
      r
  }

  def ixSelect(a: Tensor, ixs: List[Int]): Tensor = {
    val size = TH.THLongStorage_newWithData(a.payload.getSize, a.payload.getNDimension)
    // val stride = TH.THLongStorage_newWithData(a.payload.getStride, a.payload.getNDimension)
    // val mask = TH.THByteTensor_newWithSize(size, stride)
    val mask = TH.THByteTensor_new
    TH.THByteTensor_zeros(mask, size)
    ixs.foreach(i => TH.THByteTensor_set1d(mask, i, 1))

    val t = TH.THFloatTensor_new
    TH.THFloatTensor_maskedSelect(t, a.payload, mask)

    TH.THByteTensor_free(mask)

    new Tensor(t)
  }

  def data(a: Tensor): Array[Float] = data(a.payload)

  def data(t: THFloatTensor): Array[Float] = {
    val p = TH.THFloatTensor_data(t)
    val pa = CFloatArray.frompointer(p)
    (0 until TH.THFloatTensor_numel(t)).map(pa.getitem).toArray
  }

}
