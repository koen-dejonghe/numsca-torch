package botkop.numsca

import java.math.BigInteger

import com.typesafe.scalalogging.LazyLogging
import torch.cpu._

import scala.language.postfixOps

class NDArray private (val payload: THFloatTensor) extends LazyLogging {

  val dim: Int = payload.getNDimension

  val shape: List[Int] = {
    val s = CInt64Array.frompointer(payload.getSize)
    (0 until dim).toList.map(i => s.getitem(i).toInt)
  }

  val size: Int = shape.product

  def desc: String = TH.THFloatTensor_desc(payload).getStr
  def numel: Int = TH.THFloatTensor_numel(payload)

  def data: Array[Float] = {
    val p = TH.THFloatTensor_data(payload)
    val a = CFloatArray.frompointer(p)
    (0 until size).map(a.getitem).toArray
  }

  /**
    * Free tensor and notify memory manager
    */
  override def finalize(): Unit = {
    val memSize = MemoryManager.dec(size) // obtain size before freeing!
    logger.debug(s"freeing (mem = $memSize)")
    payload.delete()
    // TH.THFloatTensor_free(payload)
  }

  def reshape(newShape: List[Int]): NDArray =
    NDArray.reshape(this, newShape)

  def reshape(newShape: Int*): NDArray = reshape(newShape.toList)

  def apply(i: Int*): NDArray = NDArray.select(this, i.toList)

  def :=(a: NDArray): Unit = NDArray.assign(this, a)
}

object NDArray extends LazyLogging {

  val rng: SWIGTYPE_p_THGenerator = TH.THGenerator_new()

//  def initialSeed: Long = TH.THRandom_initialSeed(rng).longValue()
//  def currentSeed: Long = TH.THRandom_seed(rng).longValue()
  def setSeed(theSeed: Long): Unit =
    TH.THRandom_manualSeed(rng, BigInteger.valueOf(theSeed))

  def create(data: Array[Float], shape: List[Int]): NDArray = {
    require(data.length == shape.product)
    MemoryManager.memCheck(shape)
    val size = data.length
    val a = floatArray(data)
    val storage: THFloatStorage = TH.THFloatStorage_newWithData(a, size)
    val t =
      TH.THFloatTensor_newWithStorage(storage, 0, longStorage(shape), null)
    new NDArray(t)
  }

  def create(data: Array[Float]): NDArray = create(data, List(data.length))

  def create(data: Float*): NDArray = create(data.toArray)

  def apply(data: Float*): NDArray = create(data: _*)

  def zeros(shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_zeros(t, ls)
    new NDArray(t)
  }

  def zeros(shape: Int*): NDArray = zeros(shape.toList)

  def ones(shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_ones(t, ls)
    new NDArray(t)
  }

  def ones(shape: Int*): NDArray = ones(shape.toList)

  def fill(f: Float, shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_fill(t, f)
    new NDArray(t)
  }

  def arange(min: Double = 0, max: Double, step: Double = 1): NDArray = {
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_arange(t, min, max, step)
    new NDArray(t)
  }

  def randn(shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_randn(t, rng, ls)
    new NDArray(t)
  }

  def randn(shape: Int*): NDArray = randn(shape.toList)

  def randint(low: Double = 0.0, high: Double, shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_uniform(t, rng, low, high)
    TH.THFloatTensor_floor(t, t)
    new NDArray(t)
  }

  def uniform(low: Double = 0.0, high: Double, shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_uniform(t, rng, low, high)
    new NDArray(t)
  }

  def linspace(start: Float, end: Float, steps: Long): NDArray = {
    MemoryManager.memCheck(steps)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_linspace(t, start, end, steps)
    new NDArray(t)
  }

  // operations that create a new tensor -----------------------------
  def cmul(t1: NDArray, t2: NDArray): NDArray = {
    val r = TH.THFloatTensor_new
    TH.THFloatTensor_cmul(r, t1.payload, t2.payload)
    MemoryManager.memCheck(TH.THFloatTensor_numel(r))
    new NDArray(r)
  }

  // utility functions -----------------------------

  private def longStorage(shape: Seq[Int]): THLongStorage = {
    val size = shape.length
    val data = new CInt64Array(size)
    var i = 0
    while (i < shape.length) {
      data.setitem(i, shape(i))
      i = i + 1
    }
    TH.THLongStorage_newWithData(data.cast(), size)
  }

  private def floatArray(data: Array[Float]): SWIGTYPE_p_float = {
    val size = data.length
    val a = new CFloatArray(size)
    var i = 0
    while (i < size) {
      a.setitem(i, data(i))
      i = i + 1
    }
    a.cast()
  }

  def reshape(a: NDArray, newShape: List[Int]): NDArray = {
    val t = TH.THFloatTensor_new()
    TH.THFloatTensor_reshape(t, a.payload, longStorage(newShape))
    new NDArray(t)
  }

  def setValue(a: NDArray, value: Float, index: List[Int]): Unit = {
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

  def getValue(a: NDArray, index: List[Int]): Float = {
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

  def select(a: NDArray, dimension: Int, sliceIndex: Int): NDArray = {
    val t = TH.THFloatTensor_newSelect(a.payload, dimension, sliceIndex)
    new NDArray(t)
  }

  def select(a: NDArray, where: List[Int]): NDArray = {
    val r = where.zipWithIndex.foldLeft(a.payload) {
      case (t, (i, d)) =>
        TH.THFloatTensor_newNarrow(t, d, i, 1)
    }

    val s = TH.THFloatTensor_new()
    TH.THFloatTensor_squeeze(s, r)

    new NDArray(s)
  }

  def assign(a: NDArray, src: NDArray): Unit = {
    val t =
      if (a.dim == src.dim)
        src.payload
      else { // broadcast
        TH.THFloatTensor_newExpand(src.payload, longStorage(a.shape))
      }
    TH.THFloatTensor_copy(a.payload, t)
  }

  def narrow(a: NDArray,
             dimension: Int,
             firstIndex: Int,
             size: Int): NDArray = {
    val t = TH.THFloatTensor_newNarrow(a.payload, dimension, firstIndex, size)
    new NDArray(t)
  }

  def linear(x: NDArray, y: NDArray, b: NDArray): NDArray = {
    val t = TH.THFloatTensor_new()
    TH.THFloatTensor_addmm(t, 1.0f, b.payload, 1.0f, x.payload, y.payload)
    new NDArray(t)
  }

}
