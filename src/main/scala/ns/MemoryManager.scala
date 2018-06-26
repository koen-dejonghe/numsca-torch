package ns

import java.util.TimerTask
import java.util.concurrent.atomic.AtomicLong

import com.typesafe.scalalogging.LazyLogging

object MemoryManager extends LazyLogging {

  val Threshold: Long = 2L * 1024L * 1024L * 1024L // 2 GB
  val FloatSize = 4
  private val hiMemMark = new AtomicLong(0)
  private val hiTensorMark = new AtomicLong(0)

  def dec(size: Long): Long = {
    val nt = hiTensorMark.decrementAndGet()
    logger.debug(s"dec - number of tensors: $nt")

    val m = hiMemMark.get()
    val s = size * FloatSize
    val t: Long = Math.max(m - s, 0L)
    hiMemMark.set(t)
    t
  }

  def inc(size: Long): Long = {
    val nt = hiTensorMark.incrementAndGet()
    logger.debug(s"inc - number of tensors: $nt")
    hiMemMark.addAndGet(size * FloatSize)
  }

  def memCheck(size: Long): Unit = {
    val level = inc(size)
//    if (level > Threshold) {
//       logger.debug(s"invoking gc ($level/$Threshold)")
//      System.gc()
//    }
  }

  def memCheck(shape: List[Int]): Unit = memCheck(shape.product)

  /*
  val t = new java.util.Timer()
  val task: TimerTask = new java.util.TimerTask {
    def run(): Unit = {

      logger.debug("running")

      val level = hiMemMark.longValue()
      if (level > Threshold) {
        logger.debug(s"invoking gc ($level/$Threshold)")
        System.gc()
      }

    }
  }
  t.schedule(task, 5000L, 5000L)
  */
  // task.cancel()
}
