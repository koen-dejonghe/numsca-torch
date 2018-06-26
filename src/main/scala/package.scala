import java.math.BigInteger

import com.typesafe.scalalogging.LazyLogging
import torch.cpu._

import scala.language.implicitConversions

/**
  * ns: numsca = numpy for scala
  */
package object ns extends LazyLogging {

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
    val t = TH.THFloatTensor_newClone(a)
    new Tensor(t)
  }

  def view(a: Tensor, shape: Shape): Tensor = {
    // todo test this
    val t = copy(a)
    val ls = longStorage(shape)
    TH.THFloatTensor_newView(t, ls)
    new Tensor(t)
  }

  def empty = new Tensor(TH.THFloatTensor_new())
  def array(data: Number*): Tensor = create(data: _*)
  def tensor(data: Number*): Tensor = create(data: _*)

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
  def create(data: Number*): Tensor = create(data.map(_.floatValue()).toArray)

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
  def onesLike(other: Tensor): Tensor = ones(other.shape)

  def fill(f: Number, shape: List[Int]): Tensor = {
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_fill(t, f.floatValue())
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

  def linspace(start: Number, end: Number, steps: Long): Tensor = {
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_linspace(t, start.floatValue(), end.floatValue(), steps)
    new Tensor(t)
  }

  //=== tensor manip =====================================================

  def reshape(a: Tensor, newShape: List[Int]): Tensor = {

    require(a.size == newShape.product)

    val ls = longStorage(newShape)

    val t = TH.THFloatTensor_newWithStorage(a.getStorage,
                                            a.getStorageOffset,
                                            ls,
                                            null)
    //     this creates a new storage tensor
    //    val t = TH.THFloatTensor_new()
    //    TH.THFloatTensor_reshape(t, a.payload, longStorage(newShape))
    new Tensor(t)
  }

  def reshape(a: Tensor, newShape: Int*): Tensor = reshape(a, newShape.toList)

  def setValue(a: Tensor, n: Number, index: List[Int]): Unit = {
    val value = n.floatValue()
    a.dim match {
      case 1 =>
        TH.THFloatTensor_set1d(a, index.head, value)
      case 2 =>
        TH.THFloatTensor_set2d(a, index.head, index(1), value)
      case 3 =>
        TH.THFloatTensor_set3d(a, index.head, index(1), index(2), value)
      case 4 =>
        TH.THFloatTensor_set4d(a,
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
        TH.THFloatTensor_get1d(a, index.head)
      case 2 =>
        TH.THFloatTensor_get2d(a, index.head, index(1))
      case 3 =>
        TH.THFloatTensor_get3d(a, index.head, index(1), index(2))
      case 4 =>
        TH.THFloatTensor_get4d(a, index.head, index(1), index(2), index(3))
    }
  }

  def select(a: Tensor, dimension: Int, sliceIndex: Int): Tensor = {
    val t = TH.THFloatTensor_newSelect(a, dimension, sliceIndex)
    new Tensor(t)
  }

  def narrow(a: Tensor, where: List[Int]): Tensor = {
    val r = where.zipWithIndex.foldLeft(a.array) {
      case (t, (i, d)) =>
        // todo probably need to free 't' before returning
        TH.THFloatTensor_newNarrow(t, d, i, 1)
    }

    val s = TH.THFloatTensor_new()
    TH.THFloatTensor_squeeze(s, r)
    new Tensor(s)
  }

  def narrow(a: Tensor, ranges: NumscaRangeSeq): Tensor = {
    val r = ranges.rs.zipWithIndex.foldLeft(a.array) {
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
      // todo probably need to free 't' before returning
    }

    val s = TH.THFloatTensor_new()
    TH.THFloatTensor_squeeze(s, r)
    new Tensor(s)
  }

  def assign(a: Tensor, f: Number): Tensor = {
    TH.THFloatTensor_fill(a, f.floatValue())
    a
  }

  def assign(a: Tensor, src: Tensor): Unit = {
    val t =
      if (a.size == src.size) {
        logger.debug("not broadcasting")
        src.array
      } else { // broadcast
        logger.debug("broadcasting")
        val ls = longStorage(a.shape)
        val ex = TH.THFloatTensor_newExpand(src, ls)
        ex
      }
    TH.THFloatTensor_copy(a, t)
  }

  def squeeze(a: Tensor): Tensor = {
    val s = TH.THFloatTensor_new()
    TH.THFloatTensor_squeeze(s, a)
    new Tensor(s)
  }

  def narrow(a: Tensor, dimension: Int, firstIndex: Int, size: Int): Tensor = {
    val t = TH.THFloatTensor_newNarrow(a, dimension, firstIndex, size)
    new Tensor(t)
  }

  def linear(x: Tensor, y: Tensor, b: Tensor): Tensor = {
    val t = TH.THFloatTensor_new()
    TH.THFloatTensor_addmm(t, 1.0f, b, 1.0f, x, y)
    new Tensor(t)
  }

  def add(t: Tensor, f: Number): Tensor =
    numberOp(TH.THFloatTensor_add, t, f)
  def add(a: Tensor, b: Tensor): Tensor =
    binOp((r, t, u) => TH.THFloatTensor_cadd(r, t, 1, u), a, b)
  def addi(t: Tensor, f: Number): Unit =
    TH.THFloatTensor_add(t, t, f.floatValue)
  def addi(a: Tensor, b: Tensor): Unit =
    TH.THFloatTensor_cadd(a, a, 1, b)

  def sub(t: Tensor, f: Number): Tensor =
    numberOp(TH.THFloatTensor_sub, t, f)
  def sub(a: Tensor, b: Tensor): Tensor =
    binOp((r, t, u) => TH.THFloatTensor_csub(r, t, 1, u), a, b)
  def subi(t: Tensor, f: Number): Unit =
    TH.THFloatTensor_sub(t, t, f.floatValue)
  def subi(a: Tensor, b: Tensor): Unit =
    TH.THFloatTensor_csub(a, a, 1, b)

  def mul(t: Tensor, f: Number): Tensor =
    numberOp(TH.THFloatTensor_mul, t, f)
  def mul(t1: Tensor, t2: Tensor): Tensor =
    binOp(TH.THFloatTensor_cmul, t1, t2)
  def muli(t: Tensor, f: Number): Unit =
    TH.THFloatTensor_mul(t, t, f.floatValue)
  def muli(a: Tensor, b: Tensor): Unit =
    TH.THFloatTensor_cmul(a, a, b)

  def div(t: Tensor, f: Number): Tensor =
    numberOp(TH.THFloatTensor_div, t, f)
  def div(t1: Tensor, t2: Tensor): Tensor =
    binOp(TH.THFloatTensor_cdiv, t1, t2)
  def divi(t: Tensor, f: Number): Unit =
    TH.THFloatTensor_div(t, t, f.floatValue)
  def divi(a: Tensor, b: Tensor): Unit =
    TH.THFloatTensor_cdiv(a, a, b)

  def pow(t: Tensor, f: Number): Tensor =
    numberOp(TH.THFloatTensor_pow, t, f)
  def pow(t1: Tensor, t2: Tensor): Tensor =
    binOp(TH.THFloatTensor_cpow, t1, t2)
  def powi(t: Tensor, f: Number): Unit =
    TH.THFloatTensor_pow(t, t, f.floatValue)
  def powi(a: Tensor, b: Tensor): Unit =
    TH.THFloatTensor_cpow(a, a, b)

  def equal(a: Tensor, b: Tensor): Boolean =
    TH.THFloatTensor_equal(a, b) == 1

  def eq(a: Tensor, b: Tensor): Tensor =
    booleanBinOp(TH.THFloatTensor_eqTensor, a, b)
  def ne(a: Tensor, b: Tensor): Tensor =
    booleanBinOp(TH.THFloatTensor_neTensor, a, b)
  def gt(a: Tensor, b: Tensor): Tensor =
    booleanBinOp(TH.THFloatTensor_gtTensor, a, b)
  def lt(a: Tensor, b: Tensor): Tensor =
    booleanBinOp(TH.THFloatTensor_ltTensor, a, b)
  def ge(a: Tensor, b: Tensor): Tensor =
    booleanBinOp(TH.THFloatTensor_geTensor, a, b)
  def le(a: Tensor, b: Tensor): Tensor =
    booleanBinOp(TH.THFloatTensor_leTensor, a, b)

  def neg(a: Tensor): Tensor = oneOp(TH.THFloatTensor_neg, a)
  def sqrt(a: Tensor): Tensor = oneOp(TH.THFloatTensor_sqrt, a)
  def square(a: Tensor): Tensor =
    oneOp((r, t) => TH.THFloatTensor_pow(r, t, 2), a)

  //==================================================
  def sum(a: Tensor, axis: Int, keepDim: Boolean = true): Tensor = {
    val r = TH.THFloatTensor_new()
    val nAxis = if (axis < 0) a.dim + axis else axis
    TH.THFloatTensor_sum(r, a, nAxis, if (keepDim) 1 else 0)
    new Tensor(r)
  }

  def sum(a: Tensor): Double = TH.THFloatTensor_sumall(a)

  def argmin(a: Tensor, axis: Int, keepDim: Boolean = true): Tensor = {
    val values = TH.THFloatTensor_new()
    val indices = TH.THLongTensor_new()
    TH.THFloatTensor_min(values, indices, a, axis, if (keepDim) 1 else 0)
    val t = longTensorToFloatTensor(indices)
    TH.THFloatTensor_free(values)
    TH.THLongTensor_free(indices)
    new Tensor(t)
  }

  def argmax(a: Tensor, axis: Int, keepDim: Boolean = true): Tensor = {
    val values = TH.THFloatTensor_new()
    val indices = TH.THLongTensor_new()
    TH.THFloatTensor_max(values, indices, a, axis, if (keepDim) 1 else 0)
    val t = longTensorToFloatTensor(indices)
    TH.THFloatTensor_free(values)
    TH.THLongTensor_free(indices)
    new Tensor(t)
  }

  def expandNd(ts: Seq[Tensor]): Seq[Tensor] =
    if (ts.tail.forall(_.shape == ts.head.shape)) {
      ts
    } else {
      val original = TH.new_CFloatTensorArray(ts.length)
      ts.indices.foreach { i =>
        TH.CFloatTensorArray_setitem(original, i, ts(i))
      }

      val results = TH.new_CFloatTensorArray(ts.length)
      ts.indices.foreach { i =>
        val t = TH.THFloatTensor_new()
        TH.CFloatTensorArray_setitem(results, i, t)
      }

      TH.THFloatTensor_expandNd(results, original, ts.length)
      val resized = ts.indices.foldLeft(Seq.empty[Tensor]) {
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
    val index = floatTensorToLongTensor(ix)

    val r = TH.THFloatTensor_new()
    TH.THFloatTensor_indexSelect(r, a, dim, index)
    val result = new Tensor(r)

    TH.THLongTensor_free(index)
    result
  }

  def indexSelect(a: Tensor, ixs: Seq[Tensor]): Tensor =
    ixs.indices.foldLeft(a) {
      case (acc, i) =>
        val r = indexSelect(acc, 0, ixs(i))
        r
    }

  def ixSelect(a: Tensor, ixs: List[Int]): Tensor = {
    val size =
      TH.THLongStorage_newWithData(a.getSize, a.getNDimension)
    val mask = TH.THByteTensor_new
    TH.THByteTensor_zeros(mask, size)
    ixs.foreach(i => TH.THByteTensor_set1d(mask, i, 1))

    val t = TH.THFloatTensor_new
    TH.THFloatTensor_maskedSelect(t, a, mask)

    TH.THByteTensor_free(mask)

    new Tensor(t)
  }

  def data(a: Tensor): Array[Float] = data(a.array)

  def data(t: THFloatTensor): Array[Float] = {
    val p = TH.THFloatTensor_data(t)
    val pa = CFloatArray.frompointer(p)
    (0 until TH.THFloatTensor_numel(t)).map(pa.getitem).toArray
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

  def floatTensorToLongTensor(ft: THFloatTensor): THLongTensor = {
    val lt = longStorageLike(ft)
    TH.THLongTensor_copyFloat(lt, ft)
    lt
  }

  def byteTensorToFloatTensor(bt: THByteTensor): THFloatTensor = {
    val lt = floatStorageLike(bt)
    TH.THFloatTensor_copyByte(lt, bt)
    lt
  }

  def longTensorToFloatTensor(t: THLongTensor): THFloatTensor = {
    val lt = floatStorageLike(t)
    TH.THFloatTensor_copyLong(lt, t)
    lt
  }

  def longStorageLike(t: THFloatTensor): THLongTensor =
    TH.THLongTensor_newWithSize(longStorage(shape(t)), longStorage(stride(t)))

  def floatStorageLike(t: THByteTensor): THFloatTensor =
    TH.THFloatTensor_newWithSize(longStorage(shape(t)), longStorage(stride(t)))
  def floatStorageLike(t: THLongTensor): THFloatTensor =
    TH.THFloatTensor_newWithSize(longStorage(shape(t)), longStorage(stride(t)))

  def shape(array: THFloatTensor): List[Int] =
    shapify(array.getSize, array.getNDimension)
  def stride(array: THFloatTensor): List[Int] =
    shapify(array.getStride, array.getNDimension)

  def shape(array: THByteTensor): List[Int] =
    shapify(array.getSize, array.getNDimension)
  def stride(array: THByteTensor): List[Int] =
    shapify(array.getStride, array.getNDimension)

  def shape(array: THLongTensor): List[Int] =
    shapify(array.getSize, array.getNDimension)
  def stride(array: THLongTensor): List[Int] =
    shapify(array.getStride, array.getNDimension)

  def shapify(p: SWIGTYPE_p_long_long, dim: Int): List[Int] = {
    val s: CInt64Array = CInt64Array.frompointer(p)
    (0 until dim).toList.map(i => s.getitem(i).toInt)
  }

  /*
  one ops
   */
  def oneOp(f: (THFloatTensor, THFloatTensor) => Unit, a: Tensor): Tensor = {
    val r = TH.THFloatTensor_new()
    f(r, a)
    new Tensor(r)
  }

  /*
  bin ops
   */
  def binOp(f: (THFloatTensor, THFloatTensor, THFloatTensor) => Unit,
            a: Tensor,
            b: Tensor): Tensor = {
    val r = TH.THFloatTensor_new()
    val Seq(ta, tb) = expandNd(Seq(a, b))
    f(r, ta, tb)
    new Tensor(r)
  }

  def numberOp(f: (THFloatTensor, THFloatTensor, Float) => Unit,
               t: Tensor,
               n: Number): Tensor = {
    val r = TH.THFloatTensor_new()
    f(r, t, n.floatValue())
    new Tensor(r)
  }

  def booleanBinOp(f: (THByteTensor, THFloatTensor, THFloatTensor) => Unit,
                   a: Tensor,
                   b: Tensor): Tensor = {
    val bt = TH.THByteTensor_new()
    f(bt, a, b)
    val r = byteTensorToFloatTensor(bt)
    TH.THByteTensor_free(bt)
    new Tensor(r, true)
  }

}
