module.exports = function isNumeric(n) {
  return typeof n === 'number' && isFinite(n);
};
